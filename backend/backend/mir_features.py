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


# Krumhansl-Schmuckler key profiles — 12 major + 12 minor. Source: Krumhansl
# 1990, "Cognitive Foundations of Musical Pitch." Retained as the historical
# baseline / provenance; key detection now uses the Temperley+Albrecht-Shanahan
# ensemble below (raw K-S was the weakest published profile — it biases toward
# the dominant and is poor at the major/minor axis). See ADR-0004 addendum.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)

# Temperley (Kostka-Payne) key profiles — the weights music21 ships as
# `TemperleyKostkaPayne`. They separate major vs minor (the relative-key axis)
# better than raw Krumhansl-Schmuckler, especially on non-classical material.
# Used for key/mode detection below; the deep-research addendum in ADR-0004
# records the corpus evidence and the on-file A/B that motivated the swap.
_TEMPERLEY_MAJOR = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
    dtype=np.float32,
)
_TEMPERLEY_MINOR = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0],
    dtype=np.float32,
)

# Albrecht-Shanahan (2013) corpus-trained profiles. Reported as the most
# accurate single profile overall (~87%) and, crucially, the strongest on the
# MINOR mode — the exact weakness Temperley has (Temperley over-predicts the
# relative major in minor keys). We ensemble the two so their mode biases
# cancel: Temperley anchors major, Albrecht-Shanahan anchors minor.
_ALBRECHT_MAJOR = np.array(
    [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081],
    dtype=np.float32,
)
_ALBRECHT_MINOR = np.array(
    [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052],
    dtype=np.float32,
)

# Ensembled key-profile bank: (major, minor) pairs whose per-key correlations
# are averaged. Combining corpus-trained profiles reduces any single profile's
# mode bias — see ADR-0004 addendum + the deep-research report (2026-07-08).
_KEY_PROFILE_ENSEMBLE = (
    (_TEMPERLEY_MAJOR, _TEMPERLEY_MINOR),
    (_ALBRECHT_MAJOR, _ALBRECHT_MINOR),
)


def _ensemble_key_corr(cm_centered: np.ndarray, cm_denom: float, shift: int, mode: str) -> float:
    """Mean Pearson correlation of a centered chroma against every profile in the
    ensemble, for one candidate key (tonic `shift`, `mode`)."""
    slot = 0 if mode == "major" else 1
    total = 0.0
    for pair in _KEY_PROFILE_ENSEMBLE:
        prof = np.roll(pair[slot], shift).astype(np.float64)
        prof_centered = prof - prof.mean()
        prof_denom = float(np.sqrt((prof_centered ** 2).sum())) or 1.0
        total += float((cm_centered * prof_centered).sum() / (cm_denom * prof_denom))
    return total / len(_KEY_PROFILE_ENSEMBLE)


_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Display spelling by key signature — flats/sharps chosen the way a musician
# would write the key (e.g. E♭ major, not D♯ major). The raw `key` field keeps
# the ASCII sharp spelling above so the similarity comparison stays exact; these
# names are for the "your song's stats" display only.
_MAJOR_NAMES = ["C", "D♭", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]
_MINOR_NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "G♯", "A", "B♭", "B"]

# A relative major/minor pair (e.g. C minor ↔ E♭ major) shares the same pitch
# classes, so mean-chroma Krumhansl-Schmuckler routinely near-ties them and the
# winner's raw correlation overstates certainty. When the relative key lands
# within this correlation margin we flag the call ambiguous and surface both,
# rather than claiming a confident single key. See ADR-0004.
#
# Calibrated on the live pipeline (ANALYSIS_SR = 22050 Hz, 60 s cap): a known
# E♭-major upload scores C minor 0.854 vs its relative E♭ major 0.748 — a 0.105
# margin — while the next unrelated key sits 0.30 back. 0.15 catches the relative
# near-tie with headroom without reaching non-relative keys. (Chroma, and thus
# this margin, is sample-rate sensitive — calibrate against ANALYSIS_SR, not
# a native-rate decode.)
_RELATIVE_KEY_MARGIN = 0.15


def _spell(idx: int, mode: str) -> str:
    """Key-signature-correct display name, e.g. (3, 'major') -> 'E♭ major'."""
    names = _MINOR_NAMES if mode == "minor" else _MAJOR_NAMES
    return f"{names[idx % 12]} {mode}"


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
    # Display-only, additive (defaults keep old catalog dicts loading):
    key_display: str = ""      # pretty, key-signature-correct spelling, e.g. "C minor"
    key_alt: str = ""          # relative major/minor, shown when it's a near-tie
    key_ambiguous: bool = False  # winner and its relative key are within margin

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
            key_display=str(payload.get("key_display", "")),
            key_alt=str(payload.get("key_alt", "")),
            key_ambiguous=bool(payload.get("key_ambiguous", False)),
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
    # Key detection uses a DEDICATED chroma, not the chroma_cens mean above:
    # CENS is a *matching* feature whose amplitude quantization discards the
    # pitch-class magnitude that key profiles correlate against. Instead we
    # compute a CQT chroma on the HARMONIC (percussion-removed) signal with
    # tuning correction — this markedly improves relative major/minor (mode)
    # discrimination — and correlate it against Temperley profiles. See the
    # deep-research addendum in ADR-0004. Falls back to the cens mean if the
    # richer chroma can't be computed. (chroma_mean above is unchanged and
    # still feeds the harmonic-similarity criterion, so the catalog stays
    # consistent.)
    try:
        tuning = float(librosa.estimate_tuning(y=wav, sr=sr))
        y_harmonic = librosa.effects.harmonic(wav, margin=8)
        key_chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=tuning)
        km = key_chroma.mean(axis=1).astype(np.float64)
    except Exception:
        km = chroma_mean.astype(np.float64)

    cm_centered = km - km.mean()
    cm_denom = float(np.sqrt((cm_centered ** 2).sum())) or 1.0

    best_r = -1.0
    best_idx = 0
    best_mode = "major"
    for mode_label in ("major", "minor"):
        for shift in range(12):
            r = _ensemble_key_corr(cm_centered, cm_denom, shift, mode_label)
            if r > best_r:
                best_r = r
                best_idx = shift
                best_mode = mode_label

    key = _PITCH_CLASSES[best_idx]
    mode = best_mode
    # Pearson correlation ranges [-1, 1]; map to [0, 1] confidence.
    key_confidence = float(max(0.0, min(1.0, (best_r + 1.0) / 2.0)))

    # Relative-key ambiguity: C minor's relative is E♭ major (tonic +3), a major
    # key's relative minor is at tonic -3. Correlate that one profile and, if it's
    # within the margin of the winner, surface both keys instead of a false-confident
    # single answer (the most common key-detection error — see ADR-0004).
    rel_mode = "major" if best_mode == "minor" else "minor"
    rel_idx = (best_idx + 3) % 12 if best_mode == "minor" else (best_idx - 3) % 12
    rel_r = _ensemble_key_corr(cm_centered, cm_denom, rel_idx, rel_mode)
    key_ambiguous = bool((best_r - rel_r) < _RELATIVE_KEY_MARGIN)
    key_display = _spell(best_idx, best_mode)
    key_alt = _spell(rel_idx, rel_mode) if key_ambiguous else ""

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
        key_display=key_display,
        key_alt=key_alt,
        key_ambiguous=key_ambiguous,
    )
