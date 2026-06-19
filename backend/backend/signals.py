"""Bit-equal port of `quality-scorer/src/lib/signals.js`.

The verdict logic and signal thresholds live here; the React prototype keeps a
twin JS implementation so the UI's static tooltips don't have to round-trip the
backend. Parity is verified by `tests/test_scoring_parity.py`.
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict

Status = Literal["pass", "warn", "fail"]


class SignalDef(TypedDict):
    id: str
    label: str
    short: str
    critical: bool
    threshold: str
    blurb: str


SIGNALS: list[SignalDef] = [
    {
        "id": "silence",
        "label": "Silence ratio",
        "short": "SIL",
        "critical": True,
        "threshold": "fail > 35%  ·  warn > 15%",
        "blurb": (
            "Share of frames below the noise floor. High values mean dead air or "
            "a render that never produced sound."
        ),
    },
    {
        "id": "clipping",
        "label": "Clipping",
        "short": "CLIP",
        "critical": True,
        "threshold": "fail > 2%  ·  warn > 0.3%",
        "blurb": (
            "Share of samples pinned at full scale. Indicates hard digital "
            "distortion baked into the audio."
        ),
    },
    {
        "id": "noise",
        "label": "Noise · spectral flatness",
        "short": "NOISE",
        "critical": True,
        "threshold": "fail > 0.55  ·  warn > 0.40",
        "blurb": (
            "Flatness of the spectrum, 0 (tonal) → 1 (noise-like). High flatness "
            "reads as hiss, static, or a failed generation."
        ),
    },
    {
        "id": "truncation",
        "label": "Edge truncation",
        "short": "TRUNC",
        "critical": True,
        "threshold": "fail > −6 dB at an edge",
        "blurb": (
            "Energy at the very first/last frames. A loud edge means the track "
            "was cut mid-phrase rather than ending."
        ),
    },
    {
        "id": "duration",
        "label": "Duration",
        "short": "DUR",
        "critical": True,
        "threshold": "keep 20s – 360s",
        "blurb": "Total length. Stubs and runaway renders fall outside the expected band.",
    },
    {
        "id": "channel",
        "label": "Channel balance",
        "short": "CHAN",
        "critical": True,
        "threshold": "fail > 18 dB  ·  warn > 9 dB",
        "blurb": (
            "Level difference between left and right. A large gap means a dead "
            "or collapsed stereo channel."
        ),
    },
    {
        "id": "dynamics",
        "label": "Dynamic range",
        "short": "DYN",
        "critical": False,
        "threshold": "warn < 7 dB  ·  flag < 4 dB",
        "blurb": (
            "Crest factor — peak vs RMS level. Very low range reads as "
            "over-compressed or lifeless, but rarely broken on its own."
        ),
    },
]

SIGNAL_BY_ID: dict[str, SignalDef] = {s["id"]: s for s in SIGNALS}
STATUS_LABEL = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


# --- small helpers (mirror `quality-scorer/src/lib/format.js`) ----------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp01(v: float) -> float:
    return clamp(v, 0.0, 1.0)


def fmt_db(v: float, digits: int = 1) -> str:
    if v > 0:
        sign = "+"
    elif v < 0:
        sign = "−"  # typographic minus — matches JS fmtDb
    else:
        sign = "±"
    return f"{sign}{abs(v):.{digits}f}"


def fmt_duration(sec: float | int | None) -> str:
    if sec is None or (isinstance(sec, float) and math.isnan(sec)):
        return "—"
    m = int(sec // 60)
    s = round(sec % 60)
    if s == 60:                       # rollover from rounding
        m += 1
        s = 0
    return f"{m}:{s:02d}"


# --- evaluator (verbatim port of `evaluateSignal()` in signals.js) ------------

def evaluate_signal(sid: str, value: float) -> dict:
    """Map a raw measured value → {status, severity (0..1), display}."""
    if sid == "silence":
        status: Status = "fail" if value > 35 else "warn" if value > 15 else "pass"
        return {"status": status, "severity": clamp01(value / 60), "display": f"{value:.1f}%"}

    if sid == "clipping":
        status = "fail" if value > 2 else "warn" if value > 0.3 else "pass"
        return {"status": status, "severity": clamp01(value / 6), "display": f"{value:.2f}%"}

    if sid == "noise":
        status = "fail" if value > 0.55 else "warn" if value > 0.40 else "pass"
        return {"status": status, "severity": clamp01((value - 0.1) / 0.7), "display": f"{value:.2f}"}

    if sid == "truncation":
        # value is dB at the loudest edge; closer to 0 = harder cut = worse
        status = "fail" if value > -6 else "warn" if value > -18 else "pass"
        return {
            "status": status,
            "severity": clamp01((value + 42) / 42),
            "display": f"{fmt_db(value)} dB",
        }

    if sid == "duration":
        status = (
            "fail" if (value < 12 or value > 420)
            else "warn" if (value < 20 or value > 360)
            else "pass"
        )
        ideal = clamp01(min(abs(value - 20), abs(value - 360)) / 60)
        if status == "pass":
            sev = 0.0
        elif status == "warn":
            sev = 0.4 + ideal * 0.2
        else:
            sev = 0.85
        return {"status": status, "severity": sev, "display": fmt_duration(value)}

    if sid == "channel":
        status = "fail" if value > 18 else "warn" if value > 9 else "pass"
        return {"status": status, "severity": clamp01(value / 30), "display": f"{value:.1f} dB"}

    if sid == "dynamics":
        status = "fail" if value < 4 else "warn" if value < 7 else "pass"
        return {"status": status, "severity": clamp01((12 - value) / 12), "display": f"{value:.1f} dB"}

    return {"status": "pass", "severity": 0.0, "display": str(value)}
