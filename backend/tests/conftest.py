"""Shared pytest fixtures.

`parity_cases` — runs the JS `computeReport` via the real JS source and returns
its outputs alongside the raw inputs, so `test_scoring_parity.py` can compare
the Python port against the JS contract bit-for-bit.

`audio_fixtures` — synthesizes six tiny WAV files (clean / clipped / silent /
noisy / truncated / dead_channel) per session so the e2e analyzer test can
read real audio without committing binary blobs to the repo.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
PARITY_SCRIPT = REPO / "scripts" / "parity_check.mjs"

SR = 22050
LONG_S = 25.0          # within keep range (20–360 s), so duration passes
SHORT_S = 0.4          # intentionally trips duration fail + truncation fail


@pytest.fixture(scope="session")
def parity_cases() -> list[dict]:
    if not PARITY_SCRIPT.exists():
        pytest.skip(f"parity script missing: {PARITY_SCRIPT}")
    try:
        result = subprocess.run(
            ["node", str(PARITY_SCRIPT)],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        )
    except FileNotFoundError:
        pytest.skip("node is not installed")
    return json.loads(result.stdout)


@pytest.fixture(scope="session")
def audio_fixtures(tmp_path_factory) -> dict[str, Path]:
    """Synthesize one fixture per critical failure mode + a clean control."""
    root = tmp_path_factory.mktemp("audio")
    rng = np.random.default_rng(0)

    long_t = np.linspace(0, LONG_S, int(SR * LONG_S), endpoint=False, dtype=np.float32)
    fade_n = int(0.1 * SR)
    ramp = np.linspace(0, 1, fade_n, dtype=np.float32)

    def _fade_edges(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[:fade_n] *= ramp
        x[-fade_n:] *= ramp[::-1]
        return x

    # Clean: two-note chord, moderate amplitude, faded edges → all pass.
    base = 0.3 * (np.sin(2 * np.pi * 440 * long_t) + 0.6 * np.sin(2 * np.pi * 660 * long_t))
    base = _fade_edges(base.astype(np.float32))
    sf.write(root / "clean.wav", np.column_stack([base, base * 0.95]), SR)

    # Clipped: hard-clip the same signal; faded edges so truncation stays pass.
    clipped = np.clip(base * 5.0, -1.0, 1.0).astype(np.float32)
    clipped = _fade_edges(clipped)
    sf.write(root / "clipped.wav", clipped, SR)

    # Silent: near-zero everywhere → silence fail.
    silent = np.full_like(base, 1e-5)
    sf.write(root / "silent.wav", silent, SR)

    # Noisy: gaussian noise + faded edges so noise fails but truncation passes.
    noisy = (0.3 * rng.standard_normal(long_t.size).astype(np.float32))
    noisy = _fade_edges(np.clip(noisy, -1.0, 1.0))
    sf.write(root / "noisy.wav", noisy, SR)

    # Dead channel: stereo, right channel zero, full duration.
    L = base
    R = np.zeros_like(base)
    sf.write(root / "dead_channel.wav", np.column_stack([L, R]), SR)

    # Truncated: very short clip with full energy at the end (no fade-out).
    short_t = np.linspace(0, SHORT_S, int(SR * SHORT_S), endpoint=False, dtype=np.float32)
    trunc = (0.4 * np.sin(2 * np.pi * 440 * short_t)).astype(np.float32)
    sf.write(root / "truncated.wav", trunc, SR)

    return {
        "clean": root / "clean.wav",
        "clipped": root / "clipped.wav",
        "silent": root / "silent.wav",
        "noisy": root / "noisy.wav",
        "dead_channel": root / "dead_channel.wav",
        "truncated": root / "truncated.wav",
    }
