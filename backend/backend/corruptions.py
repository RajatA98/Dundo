"""Deterministic, labeled corruption recipes for building the bad-track set.

Each recipe takes a clean track's samples and returns a new array with one
named failure mode injected. The `seed` argument makes every corruption
bit-deterministic from the source track id — so re-running ingest produces
the same bad set.

Audio I/O uses soundfile's `(n,)` mono / `(n, channels)` stereo shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

CorruptionKind = Literal["silence", "clip", "truncate", "noise", "dead_channel"]
KINDS: tuple[CorruptionKind, ...] = ("silence", "clip", "truncate", "noise", "dead_channel")


def seed_from_id(track_id: str) -> int:
    """Stable FNV-1a hash of a track id → 32-bit seed."""
    h = 2166136261
    for ch in track_id.encode():
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def apply_corruption(
    samples: np.ndarray,
    sr: int,
    kind: CorruptionKind,
    *,
    seed: int,
) -> np.ndarray:
    """Apply one labeled corruption. Returns a new array; never mutates input."""
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    if n == 0:
        return samples

    if kind == "silence":
        # Zero out a 40–55% span; reliably trips silence ratio > 35%.
        out = samples.copy()
        span = int(n * rng.uniform(0.40, 0.55))
        if span > 0:
            start = int(rng.integers(0, max(1, n - span)))
            out[start:start + span] = 0
        return out

    if kind == "clip":
        # Amplify a 6–12% span by 6× and clip → ~10% of samples at full scale.
        out = samples.copy()
        span = int(n * rng.uniform(0.06, 0.12))
        if span > 0:
            start = int(rng.integers(0, max(1, n - span)))
            out[start:start + span] = np.clip(out[start:start + span] * 6.0, -1.0, 1.0)
        return out

    if kind == "truncate":
        # Slice off the last 5–15% and boost the new edge so the truncation
        # detector sees a loud cut (>−6 dB → fail).
        cut_idx = int(n * rng.uniform(0.85, 0.95))
        out = samples[:cut_idx].copy()
        edge_n = min(int(0.3 * sr), out.shape[0])
        if edge_n > 0:
            out[-edge_n:] = np.clip(out[-edge_n:] * 3.0, -1.0, 1.0)
        return out

    if kind == "noise":
        # Replace the signal with broadband noise — mimics a Suno render that
        # came out as static. Faded edges keep the truncation detector at pass
        # so the failure isolates to the noise signal.
        out = rng.normal(0, 0.35, samples.shape).astype(samples.dtype)
        out = np.clip(out, -1.0, 1.0)
        fade_n = min(int(0.5 * sr), n // 4)
        if fade_n > 0:
            ramp = np.linspace(0, 1, fade_n, dtype=out.dtype)
            if out.ndim == 1:
                out[:fade_n] *= ramp
                out[-fade_n:] *= ramp[::-1]
            else:
                out[:fade_n] *= ramp[:, None]
                out[-fade_n:] *= ramp[::-1, None]
        return out

    if kind == "dead_channel":
        # Stereo only; zero one channel → huge L/R imbalance.
        if samples.ndim == 1:
            return samples.copy()
        out = samples.copy()
        which = int(rng.integers(0, 2))
        out[:, which] = 0
        return out

    raise ValueError(f"unknown corruption kind: {kind!r}")


def corrupt_file(
    src: Path, dst: Path, kind: CorruptionKind, *, seed: int | None = None
) -> dict:
    """Read `src`, corrupt with `kind`, write to `dst`. Returns a manifest entry."""
    data, sr = sf.read(str(src), dtype="float32")
    s = seed if seed is not None else seed_from_id(f"{src.stem}__{kind}")
    out = apply_corruption(data, sr, kind, seed=s)
    sf.write(str(dst), out, sr)
    return {"source_id": src.stem, "kind": kind, "seed": s, "out_path": str(dst)}
