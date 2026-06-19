"""Phase 2 — /neighbors endpoint + similarity module tests.

Tests codify PROJECT_PLAN Phase 2 acceptance criteria. These are how Codex's
integration of `clap_windowed` + `similarity` into `api.py` gets verified.

Fast tests do NOT load CLAP; they construct fake embeddings and exercise the
`similarity` module + the `/neighbors` response shape via a TestClient.
The single CLAP-loading round-trip test is marked `slow`.

Required files in `quality-scorer/public/corpus/` for the full-stack tests:
  - corpus.json
  - embeddings.npy
  - segment_embeddings.npz
  - manifest.json

If those files are absent, the corpus-dependent tests skip with a clear reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "quality-scorer" / "public" / "corpus"


def _corpus_present() -> bool:
    return all(
        (CORPUS_DIR / name).exists()
        for name in ("corpus.json", "embeddings.npy", "segment_embeddings.npz", "manifest.json")
    )


corpus_skip = pytest.mark.skipif(
    not _corpus_present(),
    reason="corpus files not yet generated; run `python -m backend.scripts.rebuild_corpus`",
)


# ----------------------------------------------------------------------------
# similarity module — fast unit tests with synthetic embeddings
# ----------------------------------------------------------------------------


def _make_synthetic_catalog(n_tracks: int = 5, embed_dim: int = 16, rng_seed: int = 0):
    """Build (catalog_tracks, embeddings, segment_embeddings) with deterministic L2-norm rows.

    Returns the same shape the api.py startup builds in production, but tiny
    so the tests are fast and deterministic.
    """
    rng = np.random.default_rng(rng_seed)
    catalog_tracks = [{"track_id": f"t{i}", "title": f"Track {i}", "artist": "A"} for i in range(n_tracks)]

    means = rng.standard_normal((n_tracks, embed_dim)).astype(np.float32)
    means /= np.linalg.norm(means, axis=1, keepdims=True)

    # Per-track segment counts: 1, 2, 3, 1, 2 = 9 total segments
    seg_counts = [1, 2, 3, 1, 2][:n_tracks]
    segment_embeddings: dict[str, np.ndarray] = {}
    for i, cnt in enumerate(seg_counts):
        s = rng.standard_normal((cnt, embed_dim)).astype(np.float32)
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        segment_embeddings[f"t{i}"] = s

    return catalog_tracks, means, segment_embeddings


def test_build_flat_catalog_shapes_and_alignment():
    from backend.similarity import build_flat_catalog

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)

    assert cat.track_ids == ["t0", "t1", "t2", "t3", "t4"]
    assert cat.means.shape == (5, 16)
    assert cat.means.dtype == np.float32
    # 1 + 2 + 3 + 1 + 2 = 9 segments
    assert cat.segs_flat.shape == (9, 16)
    assert cat.segs_flat.dtype == np.float32
    # seg_ranges: [(0,1), (1,3), (3,6), (6,7), (7,9)]
    assert cat.seg_ranges == [(0, 1), (1, 3), (3, 6), (6, 7), (7, 9)]


def test_build_flat_catalog_preserves_row_alignment():
    """Row 0 of means and segs_flat[seg_ranges[0]] both belong to track_ids[0]."""
    from backend.similarity import build_flat_catalog

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)

    # Track 0 has exactly 1 segment. catalog row 0 of segs_flat must equal
    # the segment we stored for "t0".
    np.testing.assert_array_equal(cat.segs_flat[0], segs["t0"][0])
    # Track 1 has 2 segments at rows 1, 2.
    np.testing.assert_array_equal(cat.segs_flat[1:3], segs["t1"])


def test_build_flat_catalog_raises_on_length_mismatch():
    from backend.similarity import build_flat_catalog

    tracks, means, segs = _make_synthetic_catalog()
    means_too_short = means[:3]
    with pytest.raises(ValueError):
        build_flat_catalog(tracks, means_too_short, segs)


def test_build_flat_catalog_raises_when_segments_missing_for_a_track():
    from backend.similarity import build_flat_catalog

    tracks, means, segs = _make_synthetic_catalog()
    del segs["t2"]
    with pytest.raises(ValueError):
        build_flat_catalog(tracks, means, segs)


def test_top_k_neighbors_returns_both_similarity_metrics_per_neighbor():
    from backend.similarity import build_flat_catalog, top_k_neighbors

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)

    # Query = t2's mean and its 3 segments → cosine to t2 should be ~1.0
    query_mean = means[2].copy()
    query_segs = segs["t2"].copy()

    out = top_k_neighbors(query_mean, query_segs, cat, k=3)

    assert len(out) == 3
    for nb in out:
        assert set(nb.keys()) >= {"trackId", "meanPooledSimilarity", "maxSegmentSimilarity"}
        assert isinstance(nb["trackId"], str)
        assert isinstance(nb["meanPooledSimilarity"], float)
        assert isinstance(nb["maxSegmentSimilarity"], float)


def test_top_k_neighbors_returns_self_top_when_query_matches_track():
    """Query identical to track t2 → t2 should rank #1 with meanPooledSimilarity ≈ 1.0."""
    from backend.similarity import build_flat_catalog, top_k_neighbors

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)

    query_mean = means[2].copy()
    query_segs = segs["t2"].copy()
    out = top_k_neighbors(query_mean, query_segs, cat, k=1)

    assert out[0]["trackId"] == "t2"
    assert out[0]["meanPooledSimilarity"] == pytest.approx(1.0, abs=1e-5)
    assert out[0]["maxSegmentSimilarity"] == pytest.approx(1.0, abs=1e-5)


def test_top_k_neighbors_sorted_descending_by_mean_pooled():
    """Ranking is by meanPooledSimilarity only — even if maxSegment would re-order."""
    from backend.similarity import build_flat_catalog, top_k_neighbors

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)

    rng = np.random.default_rng(7)
    query_mean = rng.standard_normal(16).astype(np.float32)
    query_mean /= np.linalg.norm(query_mean)
    query_segs = rng.standard_normal((4, 16)).astype(np.float32)
    query_segs /= np.linalg.norm(query_segs, axis=1, keepdims=True)

    out = top_k_neighbors(query_mean, query_segs, cat, k=5)
    pooled = [nb["meanPooledSimilarity"] for nb in out]
    assert pooled == sorted(pooled, reverse=True), f"pooled not desc: {pooled}"


def test_top_k_neighbors_max_segment_is_max_over_all_pairs():
    """maxSegmentSimilarity is max over (query_window i × catalog_window j) pairs for that track."""
    from backend.similarity import build_flat_catalog, top_k_neighbors

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)

    rng = np.random.default_rng(11)
    query_mean = rng.standard_normal(16).astype(np.float32)
    query_mean /= np.linalg.norm(query_mean)
    query_segs = rng.standard_normal((3, 16)).astype(np.float32)
    query_segs /= np.linalg.norm(query_segs, axis=1, keepdims=True)

    out = top_k_neighbors(query_mean, query_segs, cat, k=5)

    # Independently compute max-segment per track and verify.
    for nb in out:
        tid = nb["trackId"]
        catalog_track_segs = segs[tid]
        full = query_segs @ catalog_track_segs.T  # (Q, Wc)
        expected_max = float(full.max())
        assert nb["maxSegmentSimilarity"] == pytest.approx(expected_max, abs=1e-5)


def test_top_k_neighbors_clamps_k_to_catalog_size():
    from backend.similarity import build_flat_catalog, top_k_neighbors

    tracks, means, segs = _make_synthetic_catalog()
    cat = build_flat_catalog(tracks, means, segs)
    query_mean = means[0].copy()
    query_segs = segs["t0"].copy()
    out = top_k_neighbors(query_mean, query_segs, cat, k=999)
    assert len(out) == 5


# ----------------------------------------------------------------------------
# threshold_from_manifest
# ----------------------------------------------------------------------------


def test_threshold_from_manifest_returns_locked_default():
    from backend.similarity import threshold_from_manifest

    manifest = {"threshold_default": 0.70, "embedding_dim": 512}
    assert threshold_from_manifest(manifest) == 0.70


def test_threshold_from_manifest_raises_when_missing():
    from backend.similarity import threshold_from_manifest

    with pytest.raises(KeyError):
        threshold_from_manifest({"embedding_dim": 512})


# ----------------------------------------------------------------------------
# /neighbors endpoint shape — uses the live FastAPI TestClient
# ----------------------------------------------------------------------------


@corpus_skip
def test_neighbors_response_includes_model_sha_and_threshold_default():
    """Top-level response must expose `modelSha` and `thresholdDefault`."""
    from fastapi.testclient import TestClient

    from backend.api import app

    with TestClient(app) as client:
        # Use any small audio file in the test fixtures; the test exists to
        # verify the response shape, not the embedding correctness.
        fixture = REPO_ROOT / "backend" / "tests" / "fixtures" / "tiny.mp3"
        if not fixture.exists():
            pytest.skip("backend/tests/fixtures/tiny.mp3 not present; create one for end-to-end tests")
        with fixture.open("rb") as f:
            r = client.post("/neighbors", files={"file": ("tiny.mp3", f, "audio/mpeg")})

    assert r.status_code == 200, r.text
    data = r.json()

    assert "modelSha" in data, "response must include modelSha at top level"
    assert "thresholdDefault" in data, "response must include thresholdDefault"
    assert isinstance(data["thresholdDefault"], float)
    assert "neighbors" in data
    assert isinstance(data["neighbors"], list)


@corpus_skip
def test_neighbors_response_has_both_similarity_metrics_per_neighbor():
    from fastapi.testclient import TestClient

    from backend.api import app

    with TestClient(app) as client:
        fixture = REPO_ROOT / "backend" / "tests" / "fixtures" / "tiny.mp3"
        if not fixture.exists():
            pytest.skip("backend/tests/fixtures/tiny.mp3 not present; create one for end-to-end tests")
        with fixture.open("rb") as f:
            r = client.post("/neighbors", files={"file": ("tiny.mp3", f, "audio/mpeg")})

    data = r.json()
    assert data["neighbors"], "expected at least one neighbor with a populated corpus"
    for nb in data["neighbors"]:
        assert "meanPooledSimilarity" in nb
        assert "maxSegmentSimilarity" in nb
        # Old `similarity` key from Phase 1 backend should be GONE — fail loudly
        # if Codex left it behind to avoid a silent dual-shape API.
        assert "similarity" not in nb, (
            "remove the legacy `similarity` field; UI consumes meanPooledSimilarity"
        )


@corpus_skip
def test_neighbors_response_neighbors_sorted_by_mean_pooled():
    from fastapi.testclient import TestClient

    from backend.api import app

    with TestClient(app) as client:
        fixture = REPO_ROOT / "backend" / "tests" / "fixtures" / "tiny.mp3"
        if not fixture.exists():
            pytest.skip("backend/tests/fixtures/tiny.mp3 not present; create one for end-to-end tests")
        with fixture.open("rb") as f:
            r = client.post("/neighbors", files={"file": ("tiny.mp3", f, "audio/mpeg")})

    pooled = [nb["meanPooledSimilarity"] for nb in r.json()["neighbors"]]
    assert pooled == sorted(pooled, reverse=True)


# ----------------------------------------------------------------------------
# Backwards-compat — /analyze still returns the legacy 7-signal shape
# ----------------------------------------------------------------------------


@corpus_skip
def test_analyze_endpoint_still_returns_legacy_shape():
    """Phase 2 must not break /analyze — Phase 3's quality badge depends on it."""
    from fastapi.testclient import TestClient

    from backend.api import app

    with TestClient(app) as client:
        fixture = REPO_ROOT / "backend" / "tests" / "fixtures" / "tiny.mp3"
        if not fixture.exists():
            pytest.skip("backend/tests/fixtures/tiny.mp3 not present; create one for end-to-end tests")
        with fixture.open("rb") as f:
            r = client.post("/analyze", files={"file": ("tiny.mp3", f, "audio/mpeg")})

    assert r.status_code == 200, r.text
    data = r.json()
    # Legacy shape — these must keep working unchanged for the quality badge.
    for k in ("score", "verdict", "reason", "signals", "waveform", "problems"):
        assert k in data, f"/analyze legacy shape missing key: {k}"


# ----------------------------------------------------------------------------
# Slow — full CLAP roundtrip
# ----------------------------------------------------------------------------


@pytest.mark.slow
@corpus_skip
def test_neighbors_top_match_is_a_known_catalog_track_when_query_is_one():
    """If we re-upload a Tier-1 catalog preview, that exact track should rank #1."""
    from fastapi.testclient import TestClient

    from backend.api import app

    # Use the first catalog track's previewUrl as the query — we don't fetch it
    # at test time (no network in CI), so this test gracefully skips if there's
    # no offline fixture for it.
    fixture = REPO_ROOT / "backend" / "tests" / "fixtures" / "tier1_self_query.mp3"
    if not fixture.exists():
        pytest.skip("backend/tests/fixtures/tier1_self_query.mp3 not present")

    with TestClient(app) as client:
        with fixture.open("rb") as f:
            r = client.post("/neighbors", files={"file": ("tier1.mp3", f, "audio/mpeg")})

    data = r.json()
    assert data["neighbors"][0]["meanPooledSimilarity"] > 0.85, (
        f"self-query should rank highly; got {data['neighbors'][0]['meanPooledSimilarity']}"
    )
