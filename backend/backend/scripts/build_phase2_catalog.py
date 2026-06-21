"""Build and publish the Phase 2 Jamendo corpus from local MTG audio.

Run this on the transient GPU box after downloading the MTG-Jamendo audio tar
mirror. The script is resumable through the checkpoint manifest: completed
tracks keep their ``encoded`` status, missing/bad tracks are marked with a
skip/error status, and pending tracks can be retried by rerunning the command.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from backend import clap_windowed, config
from backend.artist import aggregate_tracks_by_artist

from . import _corpus_writer as cw
from . import _jamendo_loader
from . import phase2_catalog
from .enrich_spike import EnrichmentResult, JamendoEnricher
from .rebuild_corpus import _decode_to_mono, _load_catalog_yaml, _per_artist_cap, _resolve_model_sha

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "backend" / "catalog.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "factory" / "artifacts" / "PHASE2_DRY_RUN_MANIFEST.json"
DEFAULT_OUT_DIR = REPO_ROOT / "quality-scorer" / "public" / "corpus"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

Encoder = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
Decoder = Callable[[bytes], tuple[np.ndarray, int]]


def main() -> None:
    args = _parse_args()
    catalog = _load_catalog_yaml(args.catalog)
    jamendo_config = phase2_catalog.validate_phase2_catalog_config(catalog)

    manifest = load_or_create_manifest(
        args.manifest,
        jamendo_config=jamendo_config,
        catalog=catalog,
        candidate_limit=args.candidate_limit,
    )
    phase2_catalog.write_json(args.manifest, manifest)

    audio_index = build_audio_index(args.audio_dir)
    encoded_tracks = load_existing_corpus(args.out)
    encoded_tracks.extend(
        encode_manifest_entries(
            manifest,
            audio_index=audio_index,
            manifest_path=args.manifest,
            limit=args.limit,
        )
    )

    write_dataset_artifacts(
        args.out,
        manifest,
        encoded_tracks,
        enrich=not args.skip_enrichment,
        build_faiss=not args.skip_faiss,
        faiss_index=args.faiss_index,
    )

    if args.push_to:
        phase2_catalog.publish_dataset_folder(args.out, args.push_to)
        print(f"[phase2] pushed dataset artifacts to {args.push_to}")

    summary = manifest["summary"]["byStatus"]
    print(f"[phase2] wrote {len(encoded_tracks)} encoded Jamendo track(s) to {args.out} · statuses={summary}")


def load_or_create_manifest(
    manifest_path: Path,
    *,
    jamendo_config: dict,
    catalog: dict,
    candidate_limit: int | None = None,
) -> dict:
    if manifest_path.exists():
        return phase2_catalog.read_json(manifest_path)

    candidates = _jamendo_loader.load_jamendo_tracks(**jamendo_config)
    return phase2_catalog.build_checkpoint_manifest(
        candidates,
        limit=candidate_limit,
        per_artist_cap=_per_artist_cap(catalog),
    )


def build_audio_index(audio_dir: Path) -> dict[str, Path]:
    """Scan the local MTG audio tree once and map Jamendo track ids to files."""
    if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)
    index: dict[str, Path] = {}
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        for token in _audio_tokens(path):
            index.setdefault(token, path)
    return index


def encode_manifest_entries(
    manifest: dict,
    *,
    audio_index: dict[str, Path],
    manifest_path: Path,
    limit: int | None = None,
    encoder: Encoder = clap_windowed.encode_windowed,
    decoder: Decoder = _decode_to_mono,
) -> list[cw.CorpusTrack]:
    """Encode pending manifest entries and checkpoint after every entry."""
    encoded: list[cw.CorpusTrack] = []
    pending = phase2_catalog.pending_manifest_items(manifest, limit=limit)
    for key, entry in tqdm(pending, desc="phase2 Jamendo encode"):
        track_id = str(entry.get("sourceTrackId") or "").strip()
        audio_path = resolve_audio_path(audio_index, track_id)
        if audio_path is None:
            phase2_catalog.mark_manifest_entry(
                manifest,
                key,
                phase2_catalog.STATUS_DEAD_URL,
                reason=f"audio file missing for Jamendo track {track_id}",
            )
            phase2_catalog.write_json(manifest_path, manifest)
            continue

        try:
            wav_mono, sr = decoder(audio_path.read_bytes())
            mean_pooled, segs = encoder(wav_mono, sr)
            corpus_track = corpus_track_from_manifest_entry(entry, mean_pooled, segs)
        except Exception as exc:
            status = (
                phase2_catalog.STATUS_SKIPPED_DECODE
                if _looks_like_decode_error(exc)
                else phase2_catalog.STATUS_ERROR_RETRYABLE
            )
            phase2_catalog.mark_manifest_entry(manifest, key, status, reason=f"{type(exc).__name__}: {exc}")
            phase2_catalog.write_json(manifest_path, manifest)
            continue

        encoded.append(corpus_track)
        phase2_catalog.mark_manifest_entry(
            manifest,
            key,
            phase2_catalog.STATUS_ENCODED,
            artifact={
                "audioPath": str(audio_path),
                "embeddingShape": list(np.asarray(mean_pooled).shape),
                "segmentShape": list(np.asarray(segs).shape),
            },
        )
        phase2_catalog.write_json(manifest_path, manifest)
    return encoded


def resolve_audio_path(audio_index: dict[str, Path], source_track_id: str) -> Path | None:
    for token in _track_tokens(source_track_id):
        if token in audio_index:
            return audio_index[token]
    return None


def corpus_track_from_manifest_entry(
    entry: dict,
    mean_pooled: np.ndarray,
    segs: np.ndarray,
) -> cw.CorpusTrack:
    source_track_id = str(entry.get("sourceTrackId") or "")
    return cw.CorpusTrack(
        track_id=entry.get("trackId") or f"tier2:jamendo:{source_track_id}",
        tier="tier2",
        title=entry.get("title") or f"Jamendo {source_track_id}",
        artist=entry.get("artist") or "Unknown artist",
        primary_genre=entry.get("primaryGenre"),
        source="jamendo",
        source_url=entry.get("sourceUrl") or f"https://www.jamendo.com/track/{source_track_id}",
        track_view_url=None,
        attribution_required=False,
        license_short=entry.get("license"),
        artwork_url=None,
        duration_ms=None,
        external_ids={
            "jamendoTrackId": source_track_id,
            "jamendoAudioUrl": entry.get("audioUrl"),
        },
        mean_pooled=np.asarray(mean_pooled, dtype=np.float32),
        segment_embeddings=np.asarray(segs, dtype=np.float32),
    )


def write_dataset_artifacts(
    out_dir: Path,
    checkpoint_manifest: dict,
    tracks: list[cw.CorpusTrack],
    *,
    enrich: bool = True,
    build_faiss: bool = True,
    faiss_index: str = "faiss.index",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cw.write_corpus(out_dir, tracks)
    cw.write_examples(out_dir, [])
    rows = [track_row(track) for track in tracks]
    phase2_catalog.write_json(out_dir / "artists.json", build_artists(rows, enrich=enrich))
    phase2_catalog.write_json(out_dir / "self_retrieval.json", build_self_retrieval_sample(tracks))

    embeddings = np.load(out_dir / "embeddings.npy")
    if build_faiss:
        phase2_catalog.build_faiss_flat_index(embeddings, out_dir / faiss_index)

    phase2_catalog.write_phase2_manifest(
        out_dir,
        checkpoint_manifest,
        model_id=config.CLAP_MODEL_ID,
        model_sha=_resolve_model_sha(),
        embedding_dim=config.CLAP_EMBED_DIM,
        window_seconds=config.CLAP_WINDOW_SECONDS,
        query_max_seconds=config.CLAP_QUERY_MAX_SECONDS,
        pooling=config.CLAP_POOLING,
        threshold_default=config.SIMILARITY_THRESHOLD_DEFAULT,
        tier_counts={"tier1": 0, "tier2": len(tracks)},
        generated_at=datetime.now(timezone.utc).isoformat(),
        faiss_index=faiss_index,
    )


def load_existing_corpus(out_dir: Path) -> list[cw.CorpusTrack]:
    """Load already-written encoded tracks so interrupted runs preserve output."""
    corpus_path = out_dir / "corpus.json"
    embeddings_path = out_dir / "embeddings.npy"
    segments_path = out_dir / "segment_embeddings.npz"
    if not (corpus_path.exists() and embeddings_path.exists() and segments_path.exists()):
        return []

    rows = json.loads(corpus_path.read_text())
    embeddings = np.load(embeddings_path)
    segment_npz = np.load(segments_path)
    tracks: list[cw.CorpusTrack] = []
    for idx, row in enumerate(rows):
        track_id = row["track_id"]
        tracks.append(
            cw.CorpusTrack(
                **row,
                mean_pooled=np.asarray(embeddings[idx], dtype=np.float32),
                segment_embeddings=np.asarray(segment_npz[track_id], dtype=np.float32),
            )
        )
    return tracks


def build_artists(rows: list[dict], *, enrich: bool) -> list[dict]:
    records = aggregate_tracks_by_artist(rows)
    if enrich:
        enricher = JamendoEnricher()
        for artist_id, rec in records.items():
            result = enricher.enrich(rec)
            merge_enrichment(rec, result)
            records[artist_id] = rec
    return phase2_catalog.serialize_artists(rows, records)


def merge_enrichment(rec, result: EnrichmentResult) -> None:
    if result.sourceArtistId:
        rec.sourceArtistId = result.sourceArtistId
    if result.location:
        rec.location = result.location
    if result.supportLinks:
        rec.supportLinks = result.supportLinks
        jamendo = next((link.url for link in result.supportLinks if link.kind == "jamendo"), None)
        if jamendo:
            rec.listenUrl = jamendo
    if result.spotifyUrl and result.spotifyConfidence:
        rec.spotifyUrl = result.spotifyUrl


def build_self_retrieval_sample(tracks: list[cw.CorpusTrack], *, limit: int = 5) -> list[dict]:
    if not tracks:
        return []
    means = np.stack([np.asarray(t.mean_pooled, dtype=np.float32) for t in tracks], axis=0)
    out: list[dict] = []
    for idx, track in enumerate(tracks[:limit]):
        sims = means @ np.asarray(track.mean_pooled, dtype=np.float32)
        best = int(np.argmax(sims))
        out.append(
            {
                "queryTrackId": track.track_id,
                "topTrackId": tracks[best].track_id,
                "topSimilarity": float(sims[best]),
                "passed": tracks[best].track_id == track.track_id,
            }
        )
    return out


def track_row(track: cw.CorpusTrack) -> dict:
    row = asdict(track)
    row.pop("mean_pooled", None)
    row.pop("segment_embeddings", None)
    return row


def _audio_tokens(path: Path) -> set[str]:
    stem = path.stem.casefold()
    tokens = {stem}
    for part in stem.replace("-", "_").split("_"):
        if part.isdigit():
            tokens.update(_track_tokens(part))
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        tokens.update(_track_tokens(digits))
    return tokens


def _track_tokens(track_id: str) -> set[str]:
    raw = str(track_id).strip().casefold()
    if not raw:
        return set()
    digits = raw.removeprefix("track_").lstrip("0") or "0"
    padded = digits.zfill(7) if digits.isdigit() else raw
    return {raw, digits, padded, f"track_{digits}", f"track_{padded}"}


def _looks_like_decode_error(exc: Exception) -> bool:
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    return any(marker in f"{name} {text}" for marker in ("decode", "audio", "soundfile", "librosa", "audioread"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--audio-dir", type=Path, required=True, help="local MTG-Jamendo audio root on the GPU box")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="encode at most this many pending manifest entries")
    p.add_argument("--candidate-limit", type=int, default=None, help="only used when creating a fresh manifest")
    p.add_argument("--push-to", default=None, help="optional HF Dataset repo id, e.g. RajatA98/dundo-corpus")
    p.add_argument("--faiss-index", default="faiss.index")
    p.add_argument("--skip-faiss", action="store_true", help="local smoke runs only; full GPU run should build FAISS")
    p.add_argument("--skip-enrichment", action="store_true", help="skip Jamendo API enrichment for local smoke runs")
    return p.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[phase2] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
