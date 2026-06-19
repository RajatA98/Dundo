"""Bit-equal port of `quality-scorer/src/lib/scoring.js`.

`compute_report(raw)` is the single source of truth for both the offline ingest
pipeline and the live `/analyze` endpoint. Parity vs the JS twin is asserted by
`tests/test_scoring_parity.py`.
"""

from __future__ import annotations

import math

from .signals import SIGNALS, clamp, evaluate_signal

# Critical signals weight hardest; a broken track passing is the costly error.
WEIGHT: dict[str, float] = {
    "silence": 1.0,
    "clipping": 1.0,
    "noise": 0.9,
    "truncation": 0.85,
    "channel": 0.7,
    "duration": 0.6,
    "dynamics": 0.4,
}

FAIL_PHRASE: dict[str, str] = {
    "silence": "mostly dead air",
    "clipping": "hard clipping",
    "noise": "noise-dominated spectrum",
    "truncation": "cut off mid-phrase",
    "duration": "length out of range",
    "channel": "collapsed stereo channel",
    "dynamics": "no dynamic range",
}


def _js_round(x: float) -> int:
    """JavaScript `Math.round` semantics: half rounds toward +∞.

    Python's built-in `round()` uses banker's rounding (half-to-even), which
    diverges on .5 boundaries. Composite scores are derived from continuous
    severity sums so .5 is rare, but parity demands the JS rule.
    """
    return math.floor(x + 0.5)


def compute_report(raw: dict) -> dict:
    """Map raw signal values → the full Track-report shape consumed by the UI.

    Verdict is precision-first: any CRITICAL signal failing → DROP, regardless
    of the composite score. Dynamics can fail without forcing a drop.
    """
    signals: list[dict] = []
    for s in SIGNALS:
        ev = evaluate_signal(s["id"], raw[s["id"]])
        signals.append({
            "id": s["id"],
            "label": s["label"],
            "short": s["short"],
            "critical": s["critical"],
            "threshold": s["threshold"],
            "blurb": s["blurb"],
            "value": raw[s["id"]],
            **ev,
        })

    penalty = 0.0
    for s in signals:
        penalty += (s["severity"] ** 1.4) * WEIGHT.get(s["id"], 0.5) * 27
    score = _js_round(clamp(100 - penalty, 0, 100))

    failed = [s for s in signals if s["status"] == "fail"]
    critical_fails = sorted(
        (s for s in failed if s["critical"]),
        key=lambda x: x["severity"],
        reverse=True,
    )
    verdict = "drop" if critical_fails else "keep"
    primary_fail = critical_fails[0]["id"] if verdict == "drop" else None

    if verdict == "drop":
        w = critical_fails[0]
        extra = f" · +{len(critical_fails) - 1} more" if len(critical_fails) > 1 else ""
        reason = (
            f"Dropped — {FAIL_PHRASE[w['id']]} "
            f"({w['label'].lower()} {w['display']}){extra}."
        )
    else:
        warns = [s for s in signals if s["status"] == "warn"]
        if warns:
            plural = "s" if len(warns) > 1 else ""
            reason = (
                f"Kept — within bounds; {len(warns)} signal{plural} "
                "flagged for review."
            )
        else:
            reason = "Kept — all technical signals within bounds."

    return {
        "score": score,
        "verdict": verdict,
        "primaryFail": primary_fail,
        "reason": reason,
        "signals": signals,
        "failModes": [s["id"] for s in failed],
    }
