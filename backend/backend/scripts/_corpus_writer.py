"""Write the five corpus output files for Phase 1.

Output location is fixed by LOCKED_DECISIONS:
  quality-scorer/public/corpus/

Files:
  - corpus.json           track metadata, one row per track
  - embeddings.npy        L2-normalized mean-pooled track vectors, shape (N, 512)
  - segment_embeddings.npz  per-window L2-normalized embeddings, indexed by track_id;
                            each value shape (num_windows_i, 512)
  - manifest.json         locked schema — see write_manifest() below
  - examples.json         3–5 staged precomputed query responses

Note: we NEVER write preview/audio bytes anywhere — only the embedding +
metadata + the source link are persisted. (The Tier-1/iTunes path that
originally motivated this rule was retired in the Dundo fork; v1 is CC-only.
The `tier`/`attribution_required`/`track_view_url` fields below are vestigial
from that era and Phase 2 may retire them — see REVIEW_REPORT.md Minor #3.)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from backend import config

# Default output dir relative to repo root. The script's main() resolves the
# absolute path from `__file__` so this works regardless of cwd.
DEFAULT_OUT_DIR_FROM_REPO = Path("quality-scorer/public/corpus")


@dataclass
class CorpusTrack:
    """Catalog track shape. v1 is CC-only (Jamendo); the Tier-1/iTunes fields
    below are vestigial from the PiedPiper era and always take the Tier-2/CC
    branch now (`tier="tier2"`, `attribution_required=False`, `license_short`
    + `source_url` = the Jamendo upstream link). Phase 2 may retire them.
    """

    track_id: str                     # repo-unique key, e.g. "tier2:jamendo:382"
    tier: str                         # always "tier2" in v1 (CC-only)
    title: str
    artist: str
    primary_genre: str | None
    source: str                       # "jamendo" (v1); "fma" reserved for v2
    source_url: str                   # link out to original source page
    track_view_url: str | None        # vestigial (was the iTunes deep-link)
    attribution_required: bool        # vestigial; always False in v1
    license_short: str | None         # e.g. "CC BY 4.0"
    artwork_url: str | None
    duration_ms: int | None
    external_ids: dict[str, Any] = field(default_factory=dict)
    # Set at ingest time by rebuild_corpus.py:
    mean_pooled: np.ndarray | None = None       # shape (CLAP_EMBED_DIM,), L2-norm
    segment_embeddings: np.ndarray | None = None  # shape (num_windows, CLAP_EMBED_DIM), L2-norm rows


def write_corpus(out_dir: Path, tracks: list[CorpusTrack]) -> None:
    """Write corpus.json + embeddings.npy + segment_embeddings.npz.

    - corpus.json: list-of-dicts, ordered identically to embeddings.npy rows.
    - embeddings.npy: stack of `track.mean_pooled` vectors, shape (N, 512), float32.
    - segment_embeddings.npz: keyed by `track.track_id`, each value is
      `track.segment_embeddings`. Use np.savez_compressed for size.

    Asserts that every track has `mean_pooled` and `segment_embeddings` set and
    that both are L2-normalized within float tolerance — fails loud if a row
    slipped through with raw embeddings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    mean_rows: list[np.ndarray] = []
    segment_rows: dict[str, np.ndarray] = {}

    for track in tracks:
        if track.mean_pooled is None:
            raise ValueError(f"{track.track_id}: missing mean_pooled")
        if track.segment_embeddings is None:
            raise ValueError(f"{track.track_id}: missing segment_embeddings")

        mean = np.asarray(track.mean_pooled, dtype=np.float32)
        segs = np.asarray(track.segment_embeddings, dtype=np.float32)
        if mean.shape != (config.CLAP_EMBED_DIM,):
            raise ValueError(f"{track.track_id}: bad mean shape {mean.shape}")
        if segs.ndim != 2 or segs.shape[1] != config.CLAP_EMBED_DIM or segs.shape[0] < 1:
            raise ValueError(f"{track.track_id}: bad segment shape {segs.shape}")
        if not np.isclose(np.linalg.norm(mean), 1.0, atol=1e-4):
            raise ValueError(f"{track.track_id}: mean embedding is not L2-normalized")
        if not np.allclose(np.linalg.norm(segs, axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{track.track_id}: segment embeddings are not L2-normalized")

        row = asdict(track)
        row.pop("mean_pooled", None)
        row.pop("segment_embeddings", None)
        rows.append(row)
        mean_rows.append(mean)
        segment_rows[track.track_id] = segs

    embeddings = (
        np.stack(mean_rows, axis=0).astype(np.float32)
        if mean_rows
        else np.empty((0, config.CLAP_EMBED_DIM), dtype=np.float32)
    )

    (out_dir / "corpus.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.save(out_dir / "embeddings.npy", embeddings)
    np.savez_compressed(out_dir / "segment_embeddings.npz", **segment_rows)


def write_examples(out_dir: Path, examples: list[dict]) -> None:
    """Write examples.json — 3–5 staged precomputed query responses.

    Each example dict includes:
      - id, chipLabel (UI label for the example chip)
      - title, artist (the query audio's identity)
      - meanPooledSimilarity, maxSegmentSimilarity (top-1 against catalog)
      - neighbors: list of {trackId, title, artist, meanPooledSimilarity, maxSegmentSimilarity}
      - verdictHeadline: either the percentage-string or "Completely unique — ..."
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")


def write_manifest(
    out_dir: Path,
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
) -> None:
    """Write manifest.json with the LOCKED schema.

    Every field listed in the signature is REQUIRED — Codex review locked this
    in PROJECT_PLAN Phase 1 acceptance criteria. The `sha256` field is
    computed by this function over the freshly-written corpus.json +
    embeddings.npy + segment_embeddings.npz files and added to the manifest
    before writing.

    Args:
        model_id: e.g. "laion/larger_clap_music".
        model_sha: HF revision SHA pinned in requirements / from_pretrained.
        embedding_dim: 512 (locked).
        window_seconds: 10.0 (locked).
        query_max_seconds: 90.0 (locked).
        pooling: "l2_normalized_mean" (locked string).
        threshold_default: 0.70 (provisional).
        tier_counts: {"tier1": N, "tier2": M}.
        generated_at: ISO 8601 UTC string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    file_names = ["corpus.json", "embeddings.npy", "segment_embeddings.npz", "examples.json"]
    file_hashes = {
        name: compute_sha256(out_dir / name)
        for name in sorted(file_names)
        if (out_dir / name).exists()
    }
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
        "sha256": {
            "files": file_hashes,
            "combined": combined.hexdigest(),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compute_sha256(file_path: Path) -> str:
    """Return the lowercase hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
