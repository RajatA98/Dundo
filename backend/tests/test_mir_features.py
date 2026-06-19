"""Unit tests for the four ADR-0004 MIR feature extractors + comparison helpers.

Synthetic-fixture tests only — no audio I/O, no model loading. Fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend import similarity
from backend.mir_features import MirFeatures, compute


# ---------------------------------------------------------------------------
# compute() — feature extraction shape + sanity
# ---------------------------------------------------------------------------


def test_compute_returns_mirfeatures_dataclass_with_expected_shapes() -> None:
    rng = np.random.default_rng(0)
    sr = 22050
    wav = (0.1 * rng.standard_normal(sr * 3)).astype(np.float32)  # 3 s of noise
    features = compute(wav, sr)

    assert isinstance(features, MirFeatures)
    assert isinstance(features.tempo_bpm, float)
    assert features.tempo_bpm >= 0
    assert features.key in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    assert features.mode in {"major", "minor"}
    assert 0.0 <= features.key_confidence <= 1.0
    assert len(features.chroma_mean) == 12
    assert len(features.timbre_mean) == 26


def test_compute_handles_empty_audio_gracefully() -> None:
    features = compute(np.zeros(0, dtype=np.float32), sr=22050)
    assert features.tempo_bpm == 0.0
    assert features.key == "C"
    assert features.mode == "major"
    assert features.chroma_mean == [0.0] * 12
    assert features.timbre_mean == [0.0] * 26


def test_mirfeatures_roundtrip_via_dict() -> None:
    """to_dict + from_dict is a no-op so corpus.json persistence is safe."""
    original = MirFeatures(
        tempo_bpm=120.5,
        key="A",
        mode="minor",
        key_confidence=0.81,
        chroma_mean=[1 / 12] * 12,
        timbre_mean=[0.1] * 26,
    )
    serialized = original.to_dict()
    reconstructed = MirFeatures.from_dict(serialized)
    assert reconstructed.tempo_bpm == pytest.approx(120.5)
    assert reconstructed.key == "A"
    assert reconstructed.mode == "minor"
    assert reconstructed.key_confidence == pytest.approx(0.81)
    assert reconstructed.chroma_mean == [1 / 12] * 12


# ---------------------------------------------------------------------------
# compare_tempos
# ---------------------------------------------------------------------------


def test_compare_tempos_within_3_bpm_is_same() -> None:
    result = similarity.compare_tempos(120.0, 122.0)
    assert result["agreement"] == pytest.approx(1.0)
    assert result["label"] == "same tempo"


def test_compare_tempos_8_bpm_apart() -> None:
    result = similarity.compare_tempos(120.0, 128.0)
    assert result["label"] == "8 BPM apart"
    assert 0.5 <= result["agreement"] <= 0.7


def test_compare_tempos_30_bpm_apart_low_agreement() -> None:
    result = similarity.compare_tempos(120.0, 150.0)
    assert result["label"] == "30 BPM apart"
    assert result["agreement"] < 0.4


def test_compare_tempos_handles_zero_or_missing_values() -> None:
    result = similarity.compare_tempos(0.0, 120.0)
    assert "BPM apart" in result["label"]
    assert result["agreement"] >= 0.0


# ---------------------------------------------------------------------------
# compare_keys
# ---------------------------------------------------------------------------


def test_compare_keys_exact_match() -> None:
    result = similarity.compare_keys("A", "minor", "A", "minor")
    assert result["agreement"] == 1.0
    assert result["label"] == "same key"


def test_compare_keys_relative_minor_major() -> None:
    # C major and A minor share the same pitches (relative key relationship).
    result = similarity.compare_keys("C", "major", "A", "minor")
    assert result["label"] == "relative key"
    assert result["agreement"] == pytest.approx(0.7)


def test_compare_keys_relative_major_minor() -> None:
    # A minor's relative major is C major; the inverse of the previous case.
    result = similarity.compare_keys("A", "minor", "C", "major")
    assert result["label"] == "relative key"


def test_compare_keys_fifth_apart_same_mode() -> None:
    # C major <-> G major are a perfect fifth apart, both major.
    result = similarity.compare_keys("C", "major", "G", "major")
    assert result["label"] == "fifth apart"
    assert result["agreement"] == pytest.approx(0.5)


def test_compare_keys_different() -> None:
    # C major <-> D# major: no special tonal relationship.
    result = similarity.compare_keys("C", "major", "D#", "major")
    assert result["label"] == "different key"
    assert result["agreement"] == 0.0


# ---------------------------------------------------------------------------
# compare_chroma_vectors
# ---------------------------------------------------------------------------


def test_compare_chroma_identical_is_very_similar() -> None:
    vec = [1 / 12] * 12
    result = similarity.compare_chroma_vectors(vec, vec)
    assert result["agreement"] == pytest.approx(1.0)
    assert result["label"] == "very similar chord palette"


def test_compare_chroma_orthogonal_is_different() -> None:
    """Chroma vectors with no shared energy → cosine = 0 → 'different chord palette'."""
    a = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    result = similarity.compare_chroma_vectors(a, b)
    assert result["label"] == "different chord palette"
    assert result["agreement"] == pytest.approx(0.0)


def test_compare_chroma_handles_empty_input() -> None:
    result = similarity.compare_chroma_vectors([], [1.0] * 12)
    assert result["agreement"] == 0.0


# ---------------------------------------------------------------------------
# compare_timbre_vectors
# ---------------------------------------------------------------------------


def test_compare_timbre_identical_is_very_similar() -> None:
    vec = [0.5] * 26
    result = similarity.compare_timbre_vectors(vec, vec)
    assert result["agreement"] == pytest.approx(1.0)
    assert "very similar production feel" == result["label"]


def test_compare_timbre_anti_correlated_clamps_to_zero() -> None:
    a = [1.0] * 26
    b = [-1.0] * 26
    result = similarity.compare_timbre_vectors(a, b)
    assert result["agreement"] == 0.0
    assert "different production" in result["label"]


def test_compare_timbre_orthogonal() -> None:
    a = [1.0] + [0.0] * 25
    b = [0.0] + [1.0] + [0.0] * 24
    result = similarity.compare_timbre_vectors(a, b)
    assert result["agreement"] == pytest.approx(0.0)
