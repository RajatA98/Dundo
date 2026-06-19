"""Corruption round-trip: every labeled corruption must trigger a DROP.

For each kind, the contract is:
  1. The verdict is `drop` after analysis (the detector caught it).
  2. The labeled failure signal's status is not `pass` (the corruption is
     genuinely affecting the signal we said it would).
  3. The output is bit-deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from backend.corruptions import KINDS, apply_corruption, seed_from_id
from backend.librosa_engine import analyze_array
from backend.scoring import compute_report

SIGNAL_FOR_KIND = {
    "silence": "silence",
    "clip": "clipping",
    "truncate": "truncation",
    "noise": "noise",
    "dead_channel": "channel",
}


def _analyze(samples: np.ndarray, sr: int) -> tuple[dict, dict]:
    # soundfile uses (n, channels); librosa uses (channels, n).
    y = samples.T if samples.ndim == 2 else samples
    res = analyze_array(y, sr, duration_override=samples.shape[0] / sr)
    return compute_report(res["raw"]), res["raw"]


@pytest.mark.parametrize("kind", KINDS)
def test_corruption_triggers_drop(audio_fixtures, kind: str) -> None:
    data, sr = sf.read(str(audio_fixtures["clean"]), dtype="float32")
    out = apply_corruption(data, sr, kind, seed=seed_from_id(f"clean__{kind}"))
    report, raw = _analyze(out, sr)

    assert report["verdict"] == "drop", (
        f"{kind}: expected DROP; raw={raw} reason={report['reason']!r}"
    )

    expected_signal_id = SIGNAL_FOR_KIND[kind]
    sig = next(s for s in report["signals"] if s["id"] == expected_signal_id)
    assert sig["status"] != "pass", (
        f"{kind}: expected {expected_signal_id!r} status≠pass; "
        f"got {sig['status']!r} value={sig['value']}"
    )


def test_corruption_is_deterministic(audio_fixtures) -> None:
    data, _ = sf.read(str(audio_fixtures["clean"]), dtype="float32")
    a = apply_corruption(data, 22050, "clip", seed=42)
    b = apply_corruption(data, 22050, "clip", seed=42)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "same seed must produce identical output"


def test_corruption_does_not_mutate_input(audio_fixtures) -> None:
    data, sr = sf.read(str(audio_fixtures["clean"]), dtype="float32")
    snapshot = data.copy()
    apply_corruption(data, sr, "silence", seed=1)
    apply_corruption(data, sr, "clip", seed=2)
    apply_corruption(data, sr, "noise", seed=3)
    apply_corruption(data, sr, "dead_channel", seed=4)
    assert np.array_equal(data, snapshot), "apply_corruption must not mutate its input"
