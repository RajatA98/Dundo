"""Audio analysis via librosa.

Produces the **raw** signal values that feed `compute_report()`, plus the
180-bin waveform peaks and the problem-region list the frontend's `Waveform`
component renders. A real librosa+soundfile decode in one pass; the live
endpoint and the offline ingest CLI both call into `analyze_array`.
"""

from __future__ import annotations

import io
import math
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from . import config

EPS = config.EPS


# --- public API ---------------------------------------------------------------

def analyze(path: str | Any) -> dict:
    """Decode a file from disk and analyze it. Reads true duration from header."""
    p = str(path)
    try:
        duration_full = float(sf.info(p).duration)
    except Exception:
        duration_full = None
    y, sr = librosa.load(p, sr=config.ANALYSIS_SR, mono=False)
    return analyze_array(y, sr, duration_override=duration_full)


def analyze_bytes(data: bytes) -> dict:
    """Decode raw file bytes (used by the /analyze endpoint)."""
    try:
        info = sf.info(io.BytesIO(data))
        duration_full = float(info.duration)
    except Exception:
        duration_full = None
    y, sr = librosa.load(io.BytesIO(data), sr=config.ANALYSIS_SR, mono=False)
    return analyze_array(y, sr, duration_override=duration_full)


def analyze_array(
    y: np.ndarray, sr: int, *, duration_override: float | None = None
) -> dict:
    """Compute the 7 raw signals + waveform + problem regions from samples.

    `y` may be 1-D (mono) or 2-D `(channels, n)` (stereo).
    """
    # Normalize shape & cache pre-cap arrays for the duration & truncation views.
    if y.ndim == 1:
        stereo_pre: np.ndarray | None = None
        mono_pre = y.astype(np.float32, copy=False)
    else:
        stereo_pre = y.astype(np.float32, copy=False)
        mono_pre = librosa.to_mono(stereo_pre)

    cap_n = int(config.CLIP_CAP_S * sr)
    if mono_pre.shape[-1] > cap_n:
        mono = mono_pre[:cap_n]
        stereo = stereo_pre[:, :cap_n] if stereo_pre is not None else None
    else:
        mono = mono_pre
        stereo = stereo_pre

    duration_sec = (
        float(duration_override)
        if duration_override is not None
        else float(mono_pre.shape[-1] / max(sr, 1))
    )

    raw = {
        "silence": _silence_pct(mono),
        "clipping": _clipping_pct(mono),
        "noise": _noise_flatness(mono),
        "truncation": _truncation_db(mono, sr),
        "duration": duration_sec,
        "channel": _channel_db(stereo),
        "dynamics": _dynamics_db(mono),
    }

    waveform = _waveform_peaks(mono, n_bins=config.WAVEFORM_BINS)
    problems = _problem_regions(mono, sr, raw, n_bins=config.WAVEFORM_BINS)

    # Mean RMS loudness in dBFS — an honest intensity proxy for the "energy" stat.
    # Surfaced top-level (not in `raw`) so the scoring/report path is untouched.
    rms_mean = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    loudness_db = 20.0 * math.log10(max(rms_mean, EPS))

    return {
        "raw": raw,
        "waveform": waveform,
        "problems": problems,
        "durationSec": duration_sec,
        "loudnessDb": loudness_db,
    }


# --- per-signal extractors ----------------------------------------------------

def _silence_pct(mono: np.ndarray) -> float:
    if mono.size == 0:
        return 100.0
    rms = librosa.feature.rms(
        y=mono, frame_length=config.FRAME_SIZE, hop_length=config.HOP_SIZE
    )[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, EPS))
    floor = max(float(rms_db.max()) - 50.0, -60.0)
    return float(np.mean(rms_db < floor) * 100.0)


def _clipping_pct(mono: np.ndarray) -> float:
    if mono.size == 0:
        return 0.0
    return float(np.mean(np.abs(mono) >= 0.999) * 100.0)


def _noise_flatness(mono: np.ndarray) -> float:
    if mono.size < config.FRAME_SIZE:
        return 0.0
    flat = librosa.feature.spectral_flatness(
        y=mono, n_fft=config.FRAME_SIZE, hop_length=config.HOP_SIZE
    )[0]
    return float(flat.mean())


def _truncation_db(mono: np.ndarray, sr: int) -> float:
    """Loudness in dBFS at the louder edge. Closer to 0 = harder cut = worse."""
    window_n = int(0.2 * sr)
    if window_n <= 0 or mono.shape[-1] < 2 * window_n:
        return -60.0
    head = mono[:window_n]
    tail = mono[-window_n:]
    head_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(head * head))), EPS))
    tail_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(tail * tail))), EPS))
    return float(max(head_db, tail_db))


def _channel_db(stereo: np.ndarray | None) -> float:
    if stereo is None or stereo.ndim != 2 or stereo.shape[0] < 2:
        return 0.0
    L = stereo[0]
    R = stereo[1]
    l_rms = float(np.sqrt(np.mean(L * L)))
    r_rms = float(np.sqrt(np.mean(R * R)))
    return float(abs(20.0 * math.log10(max(l_rms, EPS) / max(r_rms, EPS))))


def _dynamics_db(mono: np.ndarray) -> float:
    if mono.size == 0:
        return 0.0
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono * mono)))
    return float(20.0 * math.log10(max(peak, EPS) / max(rms, EPS)))


# --- waveform peaks (180 bins) ------------------------------------------------

def _waveform_peaks(mono: np.ndarray, *, n_bins: int) -> list[float]:
    n = mono.shape[-1]
    if n == 0:
        return [0.0] * n_bins
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    peaks = np.empty(n_bins, dtype=np.float32)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        peaks[i] = float(np.max(np.abs(mono[lo:hi]))) if hi > lo else 0.0
    m = float(peaks.max())
    if m > 0:
        peaks = peaks / m
    return peaks.tolist()


# --- problem regions (clip / silence / truncation; min run length 3) ---------

def _problem_regions(
    mono: np.ndarray, sr: int, raw: dict, *, n_bins: int
) -> list[dict]:
    regions: list[dict] = []
    n = mono.shape[-1]
    if n == 0:
        return regions

    # 1. clipping — bins whose chunk contains any sample at full scale.
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    clipped = np.zeros(n_bins, dtype=bool)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo and np.any(np.abs(mono[lo:hi]) >= 0.999):
            clipped[i] = True
    regions.extend(_runs_to_regions(clipped, "clip", min_run=3))

    # 2. silence — contiguous low-RMS runs of ≥ 0.5 s, mapped frame → bin.
    rms = librosa.feature.rms(
        y=mono, frame_length=config.FRAME_SIZE, hop_length=config.HOP_SIZE
    )[0]
    if rms.size:
        rms_db = 20.0 * np.log10(np.maximum(rms, EPS))
        floor = max(float(rms_db.max()) - 50.0, -60.0)
        frame_quiet = rms_db < floor
        min_frames = max(1, int(0.5 * sr / config.HOP_SIZE))
        long_runs = _keep_long_runs(frame_quiet, min_frames)
        bin_quiet = np.zeros(n_bins, dtype=bool)
        total = rms.size
        for f in np.where(long_runs)[0]:
            b = int(f * n_bins / max(total, 1))
            if 0 <= b < n_bins:
                bin_quiet[b] = True
        regions.extend(_runs_to_regions(bin_quiet, "silence", min_run=3))

    # 3. truncation — only flag a visual region on FAIL (>-6 dB). Warn-level
    # truncations are still surfaced via the SignalRow status; the waveform
    # highlight is reserved for actual breakage.
    if raw["truncation"] > -6:
        window_n = int(0.2 * sr)
        if n >= 2 * window_n:
            head_rms = float(np.sqrt(np.mean(mono[:window_n] ** 2)))
            tail_rms = float(np.sqrt(np.mean(mono[-window_n:] ** 2)))
            if tail_rms >= head_rms:
                regions.append({"type": "truncation", "from": n_bins - 10, "to": n_bins})
            else:
                regions.append({"type": "truncation", "from": 0, "to": 10})

    # noise has no per-bin region (it's a global property; UI surfaces via failModes).
    return regions


def _keep_long_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(flags)
    i = 0
    n = flags.size
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        if (j - i) >= min_len:
            out[i:j] = True
        i = j
    return out


def _runs_to_regions(
    flagged: np.ndarray, type_: str, *, min_run: int
) -> list[dict]:
    regions: list[dict] = []
    n = flagged.size
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        if (j - i) >= min_run:
            regions.append({"type": type_, "from": int(i), "to": int(j)})
        i = j
    return regions
