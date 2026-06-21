"""Phase 2 catalog build helpers.

This module owns the local, testable pieces of the Jamendo-only v1 catalog
build: source validation, quality-gate checkpoint manifests, artist metadata
serialization, optional FAISS index writing, and HF Dataset checksum handling.

The expensive MuQ encode is orchestrated by ``build_phase2_catalog.py``. These
helpers keep that job resumable and auditable before and after the cloud GPU run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.artist import ArtistRecord, aggregate_tracks_by_artist

STATUS_PENDING = "pending"
STATUS_ENCODED = "encoded"
STATUS_SKIPPED_LICENSE = "skipped_license"
STATUS_SKIPPED_DUPLICATE = "skipped_duplicate"
STATUS_SKIPPED_PER_ARTIST_CAP = "skipped_per_artist_cap"
STATUS_DEAD_URL = "dead_url"
STATUS_SKIPPED_DECODE = "skipped_decode"
STATUS_ERROR_RETRYABLE = "error_retryable"

VALID_STATUSES = {
    STATUS_PENDING,
    STATUS_ENCODED,
    STATUS_SKIPPED_LICENSE,
    STATUS_SKIPPED_DUPLICATE,
    STATUS_SKIPPED_PER_ARTIST_CAP,
    STATUS_DEAD_URL,
    STATUS_SKIPPED_DECODE,
    STATUS_ERROR_RETRYABLE,
}

CC_LICENSE_MARKERS = ("creative commons", "cc ", "cc-", "mtg-jamendo")


def validate_phase2_catalog_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Jamendo config, rejecting retired Phase-2 sources."""
    tier1 = config.get("tier1") or []
    if tier1:
        raise ValueError("Phase 2 catalog.yaml must not define retired tier1 entries")
    tier2 = config.get("tier2") or {}
    extra_sources = sorted(set(tier2) - {"jamendo"})
    if extra_sources:
        raise ValueError(f"Phase 2 v1 is Jamendo-only; remove sources: {', '.join(extra_sources)}")
    jamendo = tier2.get("jamendo")
    if not isinstance(jamendo, dict):
        raise ValueError("Phase 2 catalog.yaml must define tier2.jamendo")
    if int(jamendo.get("count", 0)) <= 0:
        raise ValueError("tier2.jamendo.count must be positive")
    return jamendo


def build_checkpoint_manifest(
    tracks: Iterable[Any],
    *,
    limit: int | None = None,
    per_artist_cap: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create the initial dry-run/full-run checkpoint manifest.

    Each candidate is marked ``pending`` unless a pre-encode quality gate can
    decide immediately: duplicate id/url, non-CC license, or per-artist cap.
    Runtime failures such as decode errors are recorded later with
    ``mark_manifest_entry``.
    """
    entries: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    artist_counts: dict[str, int] = defaultdict(int)

    selected = list(tracks)
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    for track in selected:
        track_id = str(_get(track, "jamendo_track_id", "track_id", default="")).strip()
        key = f"jamendo:{track_id}" if track_id else "jamendo:unknown"
        artist = str(_get(track, "artist", default="Unknown artist") or "Unknown artist")
        source_url = str(_get(track, "source_url", default="") or "")
        audio_url = str(_get(track, "audio_path_or_url", default="") or "")
        license_short = str(_get(track, "license_short", default="") or "")
        status = STATUS_PENDING
        reason = None

        if not _license_allowed(license_short):
            status, reason = STATUS_SKIPPED_LICENSE, f"license not allowlisted: {license_short}"
        elif track_id in seen_ids or source_url in seen_urls:
            status, reason = STATUS_SKIPPED_DUPLICATE, "duplicate track id or source url"
        elif per_artist_cap is not None and artist_counts[_norm_artist(artist)] >= per_artist_cap:
            status, reason = STATUS_SKIPPED_PER_ARTIST_CAP, f"per-artist cap {per_artist_cap}"

        if track_id:
            seen_ids.add(track_id)
        if source_url:
            seen_urls.add(source_url)
        if status == STATUS_PENDING:
            artist_counts[_norm_artist(artist)] += 1

        entries[key] = {
            "source": "jamendo",
            "sourceTrackId": track_id,
            "trackId": f"tier2:jamendo:{track_id}" if track_id else None,
            "artist": artist,
            "title": _get(track, "title", default=None),
            "primaryGenre": _get(track, "primary_genre", "primaryGenre", default=None),
            "sourceUrl": source_url,
            "audioUrl": audio_url,
            "license": license_short,
            "status": status,
            "reason": reason,
            "attempts": 0,
            "updatedAt": generated_at or _now(),
        }

    return {
        "schemaVersion": "phase2-manifest-v1",
        "generatedAt": generated_at or _now(),
        "source": "mtg-jamendo",
        "entries": entries,
        "summary": summarize_manifest_entries(entries),
    }


def mark_manifest_entry(
    manifest: dict[str, Any],
    key: str,
    status: str,
    *,
    reason: str | None = None,
    artifact: dict[str, Any] | None = None,
) -> None:
    """Update one checkpoint entry after encode/fetch work."""
    if status not in VALID_STATUSES:
        raise ValueError(f"unknown manifest status: {status}")
    entry = manifest["entries"][key]
    entry["status"] = status
    entry["reason"] = reason
    entry["updatedAt"] = _now()
    entry["attempts"] = int(entry.get("attempts") or 0) + 1
    if artifact:
        entry.setdefault("artifacts", {}).update(artifact)
    manifest["summary"] = summarize_manifest_entries(manifest["entries"])


def summarize_manifest_entries(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(entry.get("status") for entry in entries.values())
    return {
        "total": len(entries),
        "byStatus": {status: counts.get(status, 0) for status in sorted(VALID_STATUSES) if counts.get(status, 0)},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pending_manifest_items(
    manifest: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return pending manifest entries in stable order."""
    items = [
        (key, entry)
        for key, entry in sorted((manifest.get("entries") or {}).items())
        if entry.get("status") == STATUS_PENDING
    ]
    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


def serialize_artists(
    corpus_tracks: list[dict],
    enriched_records: dict[str, ArtistRecord] | None = None,
) -> list[dict[str, Any]]:
    """Serialize artist entities for ``artists.json``."""
    records = aggregate_tracks_by_artist(corpus_tracks)
    if enriched_records:
        for artist_id, enriched in enriched_records.items():
            if artist_id in records:
                records[artist_id] = enriched
    out: list[dict[str, Any]] = []
    for rec in sorted(records.values(), key=lambda r: r.artistId):
        out.append(
            {
                "artistId": rec.artistId,
                "name": rec.name,
                "source": rec.source,
                "trackIds": rec.trackIds,
                "representativeTrackId": rec.representativeTrackId,
                "sourceArtistId": rec.sourceArtistId,
                "listenUrl": rec.listenUrl,
                "location": rec.location,
                "supportLinks": [_support_link_to_dict(link) for link in rec.supportLinks],
                "spotifyUrl": rec.spotifyUrl,
                "previewUrl": _representative_preview_url(rec, corpus_tracks),
            }
        )
    return out


def build_faiss_flat_index(embeddings: np.ndarray, out_path: Path) -> None:
    """Write an exact FAISS IndexFlatIP over L2-normalized mean embeddings."""
    import faiss

    arr = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1)
    if arr.shape[0] and not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("FAISS catalog embeddings must be L2-normalized")
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))


def artifact_file_hashes(out_dir: Path, file_names: Iterable[str]) -> dict[str, str]:
    return {
        name: compute_file_sha256(out_dir / name)
        for name in sorted(file_names)
        if (out_dir / name).exists()
    }


def write_phase2_manifest(
    out_dir: Path,
    checkpoint_manifest: dict[str, Any],
    *,
    model_id: str,
    model_sha: str,
    embedding_dim: int,
    window_seconds: float,
    query_max_seconds: float,
    pooling: str,
    threshold_default: float,
    tier_counts: dict[str, int],
    generated_at: str,
    faiss_index: str = "faiss.index",
) -> dict[str, Any]:
    """Write the runtime manifest plus the full Phase-2 checkpoint state.

    The API still needs the Phase-1 runtime fields (model id, threshold, file
    hashes). Phase 2 additionally needs every candidate accounted for as
    encoded/skipped/dead/error, so the checkpoint entries are embedded under
    ``phase2``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    file_names = [
        "artists.json",
        "corpus.json",
        "embeddings.npy",
        "examples.json",
        faiss_index,
        "segment_embeddings.npz",
        "self_retrieval.json",
    ]
    file_hashes = artifact_file_hashes(out_dir, file_names)
    combined = hashlib.sha256()
    for name in sorted(file_hashes):
        combined.update(name.encode("utf-8"))
        combined.update(file_hashes[name].encode("ascii"))

    manifest = {
        "model_id": model_id,
        "model_sha": model_sha,
        "embedding_dim": embedding_dim,
        "window_seconds": window_seconds,
        "query_max_seconds": query_max_seconds,
        "pooling": pooling,
        "threshold_default": threshold_default,
        "tier_counts": tier_counts,
        "generated_at": generated_at,
        "faiss_index": faiss_index,
        "phase2": {
            "schemaVersion": checkpoint_manifest.get("schemaVersion"),
            "source": checkpoint_manifest.get("source"),
            "generatedAt": checkpoint_manifest.get("generatedAt"),
            "summary": checkpoint_manifest.get("summary"),
            "entries": checkpoint_manifest.get("entries") or {},
        },
        "sha256": {
            "files": file_hashes,
            "combined": combined.hexdigest(),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def publish_dataset_folder(out_dir: Path, repo_id: str, *, commit_message: str | None = None) -> str:
    """Upload the built corpus folder to a Hugging Face Dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Publish Dundo Phase 2 Jamendo corpus",
    )
    return repo_id


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest_file_hashes(corpus_dir: Path) -> None:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = (manifest.get("sha256") or {}).get("files") or {}
    missing = [name for name in expected if not (corpus_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"corpus artifact(s) missing after snapshot: {', '.join(missing)}")
    mismatches = [
        name for name, want in expected.items()
        if compute_file_sha256(corpus_dir / name) != want
    ]
    if mismatches:
        raise ValueError(f"corpus checksum mismatch: {', '.join(mismatches)}")


def _license_allowed(license_short: str) -> bool:
    lowered = license_short.casefold()
    return any(marker in lowered for marker in CC_LICENSE_MARKERS)


def _support_link_to_dict(link: Any) -> dict[str, Any]:
    if hasattr(link, "model_dump"):
        return link.model_dump()
    if hasattr(link, "dict"):
        return link.dict()
    return dict(link)


def _representative_preview_url(rec: ArtistRecord, tracks: list[dict]) -> str | None:
    by_id = {str(t.get("track_id")): t for t in tracks}
    track = by_id.get(str(rec.representativeTrackId)) if rec.representativeTrackId else None
    if not track:
        return None
    ext = track.get("external_ids") or {}
    return ext.get("jamendoAudioUrl") or ext.get("previewUrl") or track.get("previewUrl")


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return default


def _norm_artist(name: str) -> str:
    return " ".join(name.casefold().split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
