"""End-to-end: each synthesized fixture must produce the expected verdict.

This is the round-trip sanity check — does the librosa engine + the scoring
port together correctly detect each failure mode in real audio?
"""

from __future__ import annotations

import pytest

from backend.librosa_engine import analyze
from backend.scoring import compute_report


def test_clean_passes(audio_fixtures) -> None:
    r = analyze(audio_fixtures["clean"])
    report = compute_report(r["raw"])
    assert report["verdict"] == "keep", (
        f"clean track should be kept; reason={report['reason']!r} raw={r['raw']}"
    )


@pytest.mark.parametrize(
    "fixture_key, expected_fail",
    [
        ("clipped", "clipping"),
        ("silent", "silence"),
        ("noisy", "noise"),
        ("dead_channel", "channel"),
    ],
)
def test_critical_failure_detection(audio_fixtures, fixture_key, expected_fail) -> None:
    r = analyze(audio_fixtures[fixture_key])
    report = compute_report(r["raw"])
    assert report["verdict"] == "drop", (
        f"{fixture_key}: should be dropped; reason={report['reason']!r} raw={r['raw']}"
    )
    assert expected_fail in report["failModes"], (
        f"{fixture_key}: expected {expected_fail!r} in failModes={report['failModes']} raw={r['raw']}"
    )


def test_truncated_is_dropped(audio_fixtures) -> None:
    # The truncated fixture trips BOTH `duration` (0.4s < 12s) and `truncation`
    # (energy at the cut edge). Both are critical — either is an acceptable
    # primary fail, but the verdict must be DROP.
    r = analyze(audio_fixtures["truncated"])
    report = compute_report(r["raw"])
    assert report["verdict"] == "drop", f"reason={report['reason']!r} raw={r['raw']}"
    assert report["primaryFail"] in ("truncation", "duration"), report["primaryFail"]
    assert any(m in report["failModes"] for m in ("truncation", "duration"))


def test_analyze_output_shape(audio_fixtures) -> None:
    r = analyze(audio_fixtures["clean"])
    # The frontend contract: 180-bin waveform, 7 raw signals, list of problems.
    assert set(r.keys()) == {"raw", "waveform", "problems", "durationSec", "loudnessDb"}
    assert len(r["waveform"]) == 180
    assert all(0.0 <= v <= 1.0 for v in r["waveform"])
    assert set(r["raw"].keys()) == {
        "silence", "clipping", "noise", "truncation", "duration", "channel", "dynamics",
    }
    assert isinstance(r["problems"], list)
    # Clean fixture: no problem regions expected.
    assert r["problems"] == []


def test_problem_regions_present_when_clipped(audio_fixtures) -> None:
    r = analyze(audio_fixtures["clipped"])
    types = {p["type"] for p in r["problems"]}
    assert "clip" in types, f"clipping fixture missing clip regions: {r['problems']}"
