"""Classical MIR features per ADR-0004 — tempo, key+mode, chroma, MFCC.

Four locked criteria for the multi-criterion similarity layer:

  - tempo      : librosa.beat.beat_track → BPM scalar
  - key, mode  : chroma_cens mean → Krumhansl-Schmuckler 24-profile correlation
  - chroma     : chroma_cens 12-d mean vector → cosine over catalog
  - mfcc       : 13 MFCCs + their stddevs → 26-d "timbre fingerprint" → cosine

All four are computed at ingest time per catalog track (stored alongside the
MuQ-MuLan embedding) and at query time per upload. Pure NumPy + librosa, no
new dependencies. ~350 ms total per 30-second clip on CPU.

The comparison helpers (compare_tempos, compare_keys, compare_chroma_vectors,
compare_timbre_vectors) live in `similarity.py` so all per-criterion math
stays next to the existing similarity primitives.

See `docs/decisions/0004-multi-criterion-similarity.md` for the design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# Krumhansl-Schmuckler key profiles — 12 major + 12 minor, each shifted to
# a different tonic. Source: Krumhansl 1990, "Cognitive Foundations of
# Musical Pitch." These are the standard tonal-strength weights for each
# pitch class within a given key; correlating a measured chroma mean
# against each rotation finds the most-likely key.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class MirFeatures:
    """Per-track MIR feature payload.

    Stored in corpus.json under the `mir_features` key and computed at
    query time on uploads. Numeric scalars + small vectors only, JSON-safe.
    """

    tempo_bpm: float
    key: str            # e.g. "A"
    mode: str           # "major" or "minor"
    key_confidence: float  # 0-1 — Krumhansl-Schmuckler correlation strength
    chroma_mean: list   # 12-d, float, sums approximately to 1.0 (probability over pitch classes)
    timbre_mean: list   # 26-d (13 MFCC means + 13 MFCC stddevs), float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "MirFeatures":
        return cls(
            tempo_bpm=float(payload["tempo_bpm"]),
            key=str(payload["key"]),
            mode=str(payload["mode"]),
            key_confidence=float(payload.get("key_confidence", 0.0)),
            chroma_mean=[float(v) for v in payload["chroma_mean"]],
            timbre_mean=[float(v) for v in payload["timbre_mean"]],
        )


def compute(wav_mono: np.ndarray, sr: int) -> MirFeatures:
    """Run all four locked MIR features on a mono audio array.

    Args:
        wav_mono: 1-D float audio at any sample rate.
        sr:       sample rate of `wav_mono`.

    Returns:
        MirFeatures dataclass with tempo, key, mode, key_confidence,
        chroma_mean (12-d), timbre_mean (26-d).

    Cost: ~350 ms on CPU for a 30-second clip.
    """
    import librosa

    wav = np.asarray(wav_mono, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return MirFeatures(
            tempo_bpm=0.0,
            key="C",
            mode="major",
            key_confidence=0.0,
            chroma_mean=[0.0] * 12,
            timbre_mean=[0.0] * 26,
        )

    # --- tempo ----------------------------------------------------------
    # beat_track returns a scalar BPM. librosa 0.10+ returns it as a
    # 1-element ndarray; coerce to float.
    tempo_arr, _beats = librosa.beat.beat_track(y=wav, sr=sr)
    tempo_bpm = float(np.asarray(tempo_arr).flatten()[0])

    # --- chroma --------------------------------------------------------
    # chroma_cens is the smoothed CENS variant; more robust to articulation
    # and tempo variations than basic chroma_stft. 12 pitch-class energies.
    chroma = librosa.feature.chroma_cens(y=wav, sr=sr)
    chroma_mean_raw = chroma.mean(axis=1).astype(np.float32)
    # Normalize to a probability-ish distribution so downstream cosine
    # comparison is scale-invariant.
    s = float(chroma_mean_raw.sum())
    chroma_mean = chroma_mean_raw / s if s > 0 else chroma_mean_raw

    # --- key + mode + confidence ---------------------------------------
    # Krumhansl-Schmuckler: correlate chroma mean against 12 rotations of
    # the major profile and 12 of the minor profile, pick the maximum.
    cm = chroma_mean.astype(np.float64)
    cm_centered = cm - cm.mean()
    cm_denom = float(np.sqrt((cm_centered ** 2).sum())) or 1.0

    best_r = -1.0
    best_idx = 0
    best_mode = "major"
    for mode_label, profile in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        for shift in range(12):
            prof = np.roll(profile, shift).astype(np.float64)
            prof_centered = prof - prof.mean()
            prof_denom = float(np.sqrt((prof_centered ** 2).sum())) or 1.0
            r = float((cm_centered * prof_centered).sum() / (cm_denom * prof_denom))
            if r > best_r:
                best_r = r
                best_idx = shift
                best_mode = mode_label

    key = _PITCH_CLASSES[best_idx]
    mode = best_mode
    # Pearson correlation ranges [-1, 1]; map to [0, 1] confidence.
    key_confidence = float(max(0.0, min(1.0, (best_r + 1.0) / 2.0)))

    # --- MFCC (timbre fingerprint) -------------------------------------
    # 13 MFCC coefficients (standard; the 0th captures overall energy and
    # is sometimes dropped, but we keep it because the mean+std combination
    # carries useful texture information).
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    timbre_mean = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1)],
    ).astype(np.float32)

    return MirFeatures(
        tempo_bpm=tempo_bpm,
        key=key,
        mode=mode,
        key_confidence=key_confidence,
        chroma_mean=[float(v) for v in chroma_mean],
        timbre_mean=[float(v) for v in timbre_mean],
    )
