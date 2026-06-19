"""Threshold table — covers every pass/warn/fail boundary for the 7 signals."""

from __future__ import annotations

import pytest

from backend.signals import evaluate_signal


@pytest.mark.parametrize(
    "sid, value, expected_status",
    [
        # silence (fail >35, warn >15)
        ("silence", 0.0, "pass"),
        ("silence", 15.0, "pass"),
        ("silence", 16.0, "warn"),
        ("silence", 35.0, "warn"),
        ("silence", 36.0, "fail"),
        # clipping (fail >2, warn >0.3)
        ("clipping", 0.0, "pass"),
        ("clipping", 0.3, "pass"),
        ("clipping", 0.5, "warn"),
        ("clipping", 2.0, "warn"),
        ("clipping", 2.5, "fail"),
        # noise (fail >0.55, warn >0.40)
        ("noise", 0.10, "pass"),
        ("noise", 0.40, "pass"),
        ("noise", 0.45, "warn"),
        ("noise", 0.55, "warn"),
        ("noise", 0.60, "fail"),
        # truncation (fail >-6, warn >-18; closer to 0 = worse)
        ("truncation", -30.0, "pass"),
        ("truncation", -18.0, "pass"),
        ("truncation", -10.0, "warn"),
        ("truncation", -6.0, "warn"),
        ("truncation", -3.0, "fail"),
        # duration
        ("duration", 30, "pass"),
        ("duration", 18, "warn"),
        ("duration", 11, "fail"),
        ("duration", 360, "pass"),
        ("duration", 380, "warn"),
        ("duration", 500, "fail"),
        # channel (fail >18, warn >9)
        ("channel", 0.0, "pass"),
        ("channel", 9.0, "pass"),
        ("channel", 12.0, "warn"),
        ("channel", 18.0, "warn"),
        ("channel", 22.0, "fail"),
        # dynamics (fail <4, warn <7) — non-critical
        ("dynamics", 10.0, "pass"),
        ("dynamics", 7.0, "pass"),
        ("dynamics", 6.0, "warn"),
        ("dynamics", 4.0, "warn"),
        ("dynamics", 3.0, "fail"),
    ],
)
def test_status_at_thresholds(sid: str, value: float, expected_status: str) -> None:
    ev = evaluate_signal(sid, value)
    assert ev["status"] == expected_status, f"{sid}={value} → {ev['status']} (wanted {expected_status})"


def test_severity_within_unit_interval() -> None:
    # Severity must always live in [0, 1] regardless of input.
    for sid in ("silence", "clipping", "noise", "truncation", "duration", "channel", "dynamics"):
        for v in (-1000, -10, -1, 0, 0.5, 1, 10, 100, 1000):
            ev = evaluate_signal(sid, v)
            assert 0.0 <= ev["severity"] <= 1.0, f"{sid}={v} severity={ev['severity']}"


def test_unknown_signal_passes_through() -> None:
    ev = evaluate_signal("nonexistent", 42)
    assert ev["status"] == "pass"
    assert ev["severity"] == 0.0
