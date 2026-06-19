"""Windowed CLAP encoding — 10 s windows, L2-normalized mean pooling.

Shared between Phase 1 (offline catalog ingest) and Phase 2 (query-time
similarity). The contract is locked in LOCKED_DECISIONS ("Track-length
normalization (binding)" section) and PROJECT_PLAN Phase 2 acceptance criteria:

- 10 s input → exactly 1 window; pooled output equals direct single-window
  CLAP encode within float tolerance.
- 30 s input → exactly 3 windows; pooled output is the L2-normalized mean of
  the per-window vectors.
- Output vector is always L2-normalized regardless of window count.
- Returns BOTH `mean_pooled_embedding` (headline ranking) and per-window
  `segment_embeddings` so the backend can compute `maxSegmentSimilarity`
  against the catalog's segment embeddings.

This module DOES NOT load CLAP itself — it delegates per-window encoding to
`clap_engine.encode_audio()` (which already exists, handles resampling to 48 kHz,
and is L2-normalizing per window). All this module owns is the chunking +
pooling math.
"""

from __future__ import annotations

import numpy as np

# ADR-0002: encoder swapped from clap_engine → muq_engine. Module name kept
# (renaming this file widens the diff for no functional gain) but the inner
# encoder is now MuQ-MuLan.
from . import muq_engine as audio_encoder
from . import config


def chunk_audio(
    wav_mono: np.ndarray,
    sr: int,
    window_seconds: float = config.CLAP_WINDOW_SECONDS,
    max_seconds: float | None = None,
) -> list[np.ndarray]:
    """Split mono audio into non-overlapping windows of `window_seconds`.

    Trims the input to `max_seconds` first if provided (used to cap query
    inputs to 90 s before any work). Audio shorter than one window returns
    `[wav_mono]` unchanged (the encoder handles short clips fine).

    Args:
        wav_mono: 1-D mono float array.
        sr: sample rate of wav_mono.
        window_seconds: per-window duration. Defaults to the locked 10 s.
        max_seconds: optional cap on total audio considered. None = use whole input.

    Returns:
        List of 1-D float arrays, each at most `window_seconds * sr` samples.
        Any trailing remainder (<= window_seconds) becomes its own final window
        only if it is at least `MIN_TAIL_FRAC * window_seconds` long — see
        constant below — otherwise it is dropped. This keeps tiny noise tails
        out of the mean-pool.
    """
    wav = np.asarray(wav_mono, dtype=np.float32)
    if wav.ndim != 1:
        wav = np.reshape(wav, (-1,))

    if max_seconds is not None:
        max_n = int(round(float(max_seconds) * sr))
        wav = wav[:max_n]

    if wav.size == 0:
        return [wav]

    window_n = max(1, int(round(float(window_seconds) * sr)))
    if wav.shape[0] <= window_n:
        return [wav]

    chunks: list[np.ndarray] = []
    for start in range(0, wav.shape[0], window_n):
        chunk = wav[start:start + window_n]
        if chunk.shape[0] == window_n:
            chunks.append(chunk)
            continue
        if chunk.shape[0] >= int(round(MIN_TAIL_FRAC * window_n)):
            chunks.append(chunk)

    return chunks or [wav[:window_n]]


# Minimum fraction of a window to keep a trailing remainder. Tunable; 0.5 means
# we keep partial tails ≥ 5 s when window_seconds = 10. Used by `chunk_audio`.
MIN_TAIL_FRAC = 0.5


def encode_windowed(
    wav_mono: np.ndarray,
    sr: int,
    window_seconds: float = config.CLAP_WINDOW_SECONDS,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode `wav_mono` as per-window embeddings AND a pooled track vector.

    Args:
        wav_mono: 1-D mono float array.
        sr: input sample rate (any rate; `clap_engine.encode_audio` resamples
            internally to 48 kHz).
        window_seconds: per-window duration. Default: 10 s (locked).
        max_seconds: optional cap on input duration. Pass
            `config.CLAP_QUERY_MAX_SECONDS` (= 90 s) for query-side encoding;
            leave None for catalog-side encoding of a ~30 s preview.

    Returns:
        Tuple of:
          - mean_pooled (np.ndarray, shape (CLAP_EMBED_DIM,), dtype float32,
            L2-normalized): the track-level embedding used for headline ranking.
          - segment_embeddings (np.ndarray, shape (num_windows, CLAP_EMBED_DIM),
            dtype float32, each row L2-normalized): per-window embeddings used
            for `maxSegmentSimilarity` against the catalog's stored segments.

    Contract:
      - num_windows >= 1 always (single-clip case returns 1 row).
      - segment_embeddings rows are L2-normalized (norms == 1.0 within float
        tolerance).
      - mean_pooled is the L2-normalized arithmetic mean of segment_embeddings.
      - For a 10 s input, the single segment row equals the direct
        `clap_engine.encode_audio(wav, sr)` output exactly (this anchors the
        Phase 2 windowing test).
    """
    chunks = chunk_audio(wav_mono, sr, window_seconds, max_seconds)
    rows = [
        l2_normalize(audio_encoder.encode_audio(chunk, sr).astype(np.float32))
        for chunk in chunks
    ]
    segment_embeddings = np.stack(rows, axis=0).astype(np.float32)
    mean_pooled = l2_normalize(segment_embeddings.mean(axis=0)).astype(np.float32)
    return mean_pooled, segment_embeddings


def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector or batch along `axis`. Float32-safe; never divides by zero."""
    arr = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return (arr / np.maximum(norm, eps)).astype(np.float32)
