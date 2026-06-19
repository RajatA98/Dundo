"""Rebuild the Dundo reference catalog from scratch.

Phase 1 entry point. Reads `backend/catalog.yaml`, hits iTunes Search + the
chosen Tier-2 source(s), runs windowed CLAP encoding on every track, and
writes the five corpus files to `quality-scorer/public/corpus/`.

Usage (from repo root):
    pip install -e "backend/[runtime,ingest,dev]"
    python -m backend.scripts.rebuild_corpus
    # or, with explicit paths:
    python -m backend.scripts.rebuild_corpus \\
        --catalog backend/catalog.yaml \\
        --out quality-scorer/public/corpus

Re-running is safe and deterministic given the same catalog.yaml +
pinned CLAP revision. The whole job is ~3–5 min on a laptop for ~300 tracks.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import audioread
import librosa
import numpy as np
from tqdm import tqdm
import yaml

from backend import config
from backend import clap_windowed

from . import _corpus_writer as cw
from . import _fma_loader, _jamendo_loader

# This file lives at <repo>/backend/backend/scripts/rebuild_corpus.py
# parents[0]=scripts/, [1]=backend/ (package), [2]=backend/ (project), [3]=<repo>
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "backend" / "catalog.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "quality-scorer" / "public" / "corpus"


def main() -> None:
    """Top-level orchestrator. Reads args, calls the per-tier ingest, writes outputs."""
    args = _parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = _load_catalog_yaml(args.catalog)

    # --- Tier 1: iTunes-curated recognizable tracks ---------------------------
    tier1_tracks = ingest_tier1(catalog.get("tier1", []))

    # --- Tier 2: bulk-loaded CC catalogs --------------------------------------
    tier2_tracks = ingest_tier2(catalog.get("tier2", {}))

    all_tracks = tier1_tracks + tier2_tracks

    # --- Build precomputed example chips --------------------------------------
    examples = build_examples(all_tracks, catalog.get("examples", []))

    # --- Write outputs --------------------------------------------------------
    cw.write_corpus(out_dir, all_tracks)
    cw.write_examples(out_dir, examples)
    cw.write_manifest(
        out_dir,
        model_id=config.CLAP_MODEL_ID,
        model_sha=_resolve_model_sha(),
        embedding_dim=config.CLAP_EMBED_DIM,
        window_seconds=config.CLAP_WINDOW_SECONDS,
        query_max_seconds=config.CLAP_QUERY_MAX_SECONDS,
        pooling=config.CLAP_POOLING,
        threshold_default=config.SIMILARITY_THRESHOLD_DEFAULT,
        tier_counts={"tier1": len(tier1_tracks), "tier2": len(tier2_tracks)},
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    print(
        f"[rebuild_corpus] wrote {len(all_tracks)} tracks "
        f"(tier1={len(tier1_tracks)}, tier2={len(tier2_tracks)}) → {out_dir}"
    )


def ingest_tier1(tier1_entries: list[dict]) -> list[cw.CorpusTrack]:
    """Retired in the Dundo pivot.

    PiedPiper's Tier-1 ingest hit Apple's iTunes Search API for previews of
    recognizable mainstream tracks. Dundo's thesis is CC-licensed indie
    discovery, so Tier-1 is gone. The catalog.yaml `tier1:` section is
    ignored; any tier1 entries pass through this no-op and don't reach the
    corpus.

    Kept as a function (rather than removed entirely) so existing callers
    + tests continue to work and the corpus stats keep a `tier1=0` line in
    the rebuild log. See factory/artifacts/ACRCLOUD_RETIREMENT_NOTE.md for
    the parallel rationale on ACRCloud removal.
    """
    if tier1_entries:
        print(
            f"[tier1] {len(tier1_entries)} entries in catalog.yaml — ignored "
            "(Tier-1 retired in Dundo pivot; CC-licensed catalog only).",
            file=sys.stderr,
        )
    return []


def ingest_tier2(tier2_config: dict) -> list[cw.CorpusTrack]:
    """Pull breadth tracks from FMA and/or Jamendo per the catalog.yaml config.

    Args:
        tier2_config: shape like:
            {
              "fma":     {"count": 100, "genres_balanced": ["Pop", "Rock", ...]},
              "jamendo": {"count": 200, "genres_balanced": [...]}
            }
          Either key is optional.

    Returns:
        Combined list of CorpusTrack from all configured Tier-2 sources.
    """
    results: list[cw.CorpusTrack] = []

    if tier2_config.get("fma"):
        try:
            fma_tracks = _fma_loader.load_fma_tracks(**tier2_config["fma"])
            for track in tqdm(fma_tracks, desc="tier2 FMA"):
                try:
                    audio_bytes = _fma_loader.fetch_track_audio(track)
                    wav_mono, sr = _decode_to_mono(audio_bytes)
                    mean_pooled, segs = clap_windowed.encode_windowed(wav_mono, sr, max_seconds=None)
                    results.append(_fma_to_corpus(track, mean_pooled, segs))
                except Exception as e:
                    print(f"[tier2:fma] skip {track.fma_track_id}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[tier2:fma] unavailable, skipping: {e}", file=sys.stderr)

    if tier2_config.get("jamendo"):
        try:
            jamendo_tracks = _jamendo_loader.load_jamendo_tracks(**tier2_config["jamendo"])
            for track in tqdm(jamendo_tracks, desc="tier2 Jamendo"):
                try:
                    audio_bytes = _jamendo_loader.fetch_track_audio(track)
                    wav_mono, sr = _decode_to_mono(audio_bytes)
                    mean_pooled, segs = clap_windowed.encode_windowed(wav_mono, sr, max_seconds=None)
                    results.append(_jamendo_to_corpus(track, mean_pooled, segs))
                except Exception as e:
                    print(f"[tier2:jamendo] skip {track.jamendo_track_id}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[tier2:jamendo] unavailable, skipping: {e}", file=sys.stderr)

    return results


def build_examples(
    all_tracks: list[cw.CorpusTrack],
    example_specs: list[dict],
) -> list[dict]:
    """Build the staged precomputed examples for the landing page chips.

    Each example_spec from catalog.yaml looks like:
        {id: "ex_suno_pop_001", chipLabel: "Suno · Pop", query_audio: "examples/audio/...mp3"}

    For each spec, decode the query audio, run windowed encode, compute top-3
    against `all_tracks`, format with both meanPooledSimilarity and
    maxSegmentSimilarity per neighbor, and produce a verdictHeadline string
    using `config.SIMILARITY_THRESHOLD_DEFAULT` for the Case A / Case B rule.

    Phase 1 contract — graceful missing-audio handling:
        - The example audio files do NOT exist yet at Phase 1 ingest time —
          they're Suno generations that arrive during Phase 6 (eval pipeline).
        - For each spec whose `query_audio` file is missing, **log a clear
          info-level message and SKIP that spec**. Do not raise.
        - Return whatever examples were successfully built (possibly an empty
          list). Tests accept 0–5 entries in Phase 1; Phase 6 enforces 3–5.
    """
    if not all_tracks:
        return []

    examples: list[dict] = []
    for spec in example_specs:
        rel = spec.get("query_audio")
        if not rel:
            continue
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"[examples] skip missing audio: {rel}")
            continue
        try:
            wav_mono, sr = _decode_to_mono(path.read_bytes())
            query_mean, query_segments = clap_windowed.encode_windowed(wav_mono, sr, max_seconds=None)
            neighbors = _top_neighbors(query_mean, query_segments, all_tracks, k=3)
        except Exception as e:
            print(f"[examples] skip {rel}: {e}", file=sys.stderr)
            continue
        if not neighbors:
            continue

        top = neighbors[0]
        if top["meanPooledSimilarity"] >= config.SIMILARITY_THRESHOLD_DEFAULT:
            headline = f"{int(top['meanPooledSimilarity'] * 100)}% similar to {top['title']} — {top['artist']}"
        else:
            headline = "Completely unique — this track doesn't sound like anything in our reference catalog"

        examples.append(
            {
                "id": spec.get("id") or path.stem,
                "chipLabel": spec.get("chipLabel") or "Example",
                "title": spec.get("title") or path.stem,
                "artist": spec.get("artist") or "Uploaded example",
                "meanPooledSimilarity": top["meanPooledSimilarity"],
                "maxSegmentSimilarity": top["maxSegmentSimilarity"],
                "neighbors": neighbors,
                "verdictHeadline": headline,
            }
        )
    return examples


def _load_catalog_yaml(path: Path) -> dict:
    """Parse catalog.yaml. Raises on missing required keys."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    if "tier1" not in data and "tier2" not in data:
        raise ValueError(f"{path}: expected at least one of tier1 or tier2")
    data.setdefault("tier1", [])
    data.setdefault("tier2", {})
    data.setdefault("examples", [])
    return data


def _decode_to_mono(audio_bytes: bytes) -> tuple:
    """Decode arbitrary audio bytes into mono float array + sample rate.

    Returns:
        (wav_mono: np.ndarray, sr: int)
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=config.ANALYSIS_SR, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as first_error:
        try:
            y, sr = _decode_with_audioread(audio_bytes)
            if sr != config.ANALYSIS_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=config.ANALYSIS_SR)
                sr = config.ANALYSIS_SR
            return y.astype(np.float32), int(sr)
        except Exception:
            raise first_error


def _decode_with_audioread(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode AAC/M4A previews through audioread/CoreAudio when SoundFile cannot."""
    with tempfile.NamedTemporaryFile(suffix=".m4a") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with audioread.audio_open(tmp.name) as reader:
            channels = int(reader.channels)
            sr = int(reader.samplerate)
            pcm_chunks = [np.frombuffer(buf, dtype=np.int16) for buf in reader]

    if not pcm_chunks:
        return np.array([], dtype=np.float32), sr

    pcm = np.concatenate(pcm_chunks).astype(np.float32) / 32768.0
    if channels > 1:
        usable = (pcm.size // channels) * channels
        pcm = pcm[:usable].reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32), sr


def _resolve_model_sha() -> str:
    """Return the pinned HF revision SHA for the CLAP model.

    Reads from a constant or env var; falls back to querying the locally
    cached model snapshot's commit hash if no pin is configured. This is what
    gets written into manifest.json so re-runs are auditable.
    """
    env = os.getenv("DUNDO_CLAP_REVISION") or os.getenv("PIEDPIPER_CLAP_REVISION")
    if env:
        return env
    const = getattr(config, "CLAP_REVISION_SHA", None)
    if const:
        return str(const)

    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_dir = hf_home / "hub" / ("models--" + config.CLAP_MODEL_ID.replace("/", "--")) / "snapshots"
    if model_dir.exists():
        snapshots = sorted((p.name for p in model_dir.iterdir() if p.is_dir()))
        if snapshots:
            return snapshots[-1]

    print("[rebuild_corpus] warning: CLAP revision is unpinned", file=sys.stderr)
    return "unpinned"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to catalog.yaml")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for corpus files")
    return p.parse_args()


def _fma_to_corpus(track: _fma_loader.FMATrack, mean_pooled: np.ndarray, segs: np.ndarray) -> cw.CorpusTrack:
    return cw.CorpusTrack(
        track_id=f"tier2:fma:{track.fma_track_id}",
        tier="tier2",
        title=track.title,
        artist=track.artist,
        primary_genre=track.primary_genre,
        source="fma",
        source_url=track.source_url,
        track_view_url=None,
        attribution_required=False,
        license_short=track.license_short,
        artwork_url=None,
        duration_ms=None,
        external_ids={"fmaTrackId": track.fma_track_id},
        mean_pooled=mean_pooled,
        segment_embeddings=segs,
    )


def _jamendo_to_corpus(track: _jamendo_loader.JamendoTrack, mean_pooled: np.ndarray, segs: np.ndarray) -> cw.CorpusTrack:
    return cw.CorpusTrack(
        track_id=f"tier2:jamendo:{track.jamendo_track_id}",
        tier="tier2",
        title=track.title,
        artist=track.artist,
        primary_genre=track.primary_genre,
        source="jamendo",
        source_url=track.source_url,
        track_view_url=None,
        attribution_required=False,
        license_short=track.license_short,
        artwork_url=None,
        duration_ms=None,
        external_ids={"jamendoTrackId": track.jamendo_track_id},
        mean_pooled=mean_pooled,
        segment_embeddings=segs,
    )


def _top_neighbors(
    query_mean: np.ndarray,
    query_segments: np.ndarray,
    tracks: list[cw.CorpusTrack],
    k: int,
) -> list[dict]:
    rows = []
    for track in tracks:
        if track.mean_pooled is None or track.segment_embeddings is None:
            continue
        mean_sim = float(np.dot(query_mean, track.mean_pooled))
        max_sim = float(np.max(query_segments @ track.segment_embeddings.T))
        rows.append(
            {
                "trackId": track.track_id,
                "title": track.title,
                "artist": track.artist,
                "source": track.source,
                "sourceUrl": track.source_url,
                "trackViewUrl": track.track_view_url,
                "meanPooledSimilarity": mean_sim,
                "maxSegmentSimilarity": max_sim,
            }
        )
    rows.sort(key=lambda r: r["meanPooledSimilarity"], reverse=True)
    return rows[:k]


if __name__ == "__main__":
    main()
