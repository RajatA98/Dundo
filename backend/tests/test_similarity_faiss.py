"""Tests for the FAISS-backed similarity backend (ADR-0007).

The contract is: same return shape as similarity.top_k_neighbors, same
numbers within float32 noise tolerance. FAISS Flat is exact (no recall
loss), only faster at scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend import similarity, similarity_faiss


def _make_synthetic_catalog(n_tracks: int = 5, embed_dim: int = 16, rng_seed: int = 0):
    """Same helper as test_neighbors_endpoint — keeps the suites symmetric."""
    rng = np.random.default_rng(rng_seed)
    catalog_tracks = [{"track_id": f"t{i}", "title": f"Track {i}", "artist": "A"} for i in range(n_tracks)]

    means = rng.standard_normal((n_tracks, embed_dim)).astype(np.float32)
    means /= np.linalg.norm(means, axis=1, keepdims=True)

    seg_counts = [1, 2, 3, 1, 2][:n_tracks]
    segment_embeddings: dict[str, np.ndarray] = {}
    for i, cnt in enumerate(seg_counts):
        s = rng.standard_normal((cnt, embed_dim)).astype(np.float32)
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        segment_embeddings[f"t{i}"] = s

    return catalog_tracks, means, segment_embeddings


@pytest.fixture(autouse=True)
def clear_cache():
    similarity_faiss.clear_index_cache()
    yield
    similarity_faiss.clear_index_cache()


def test_faiss_matches_numpy_on_known_top_match():
    """Self-query (track t2's mean + segs) → t2 ranks #1 with cosine ≈ 1.0."""
    tracks, means, segs = _make_synthetic_catalog()
    cat = similarity.build_flat_catalog(tracks, means, segs)

    query_mean = means[2].copy()
    query_segs = segs["t2"].copy()

    out = similarity_faiss.top_k_neighbors_faiss(query_mean, query_segs, cat, k=1)
    assert out[0]["trackId"] == "t2"
    assert out[0]["meanPooledSimilarity"] == pytest.approx(1.0, abs=1e-5)
    assert out[0]["maxSegmentSimilarity"] == pytest.approx(1.0, abs=1e-5)


def test_faiss_matches_numpy_ordering():
    """Top-3 ordering from FAISS Flat must match NumPy exactly."""
    tracks, means, segs = _make_synthetic_catalog()
    cat = similarity.build_flat_catalog(tracks, means, segs)

    rng = np.random.default_rng(7)
    query_mean = rng.standard_normal(16).astype(np.float32)
    query_mean /= np.linalg.norm(query_mean)
    query_segs = rng.standard_normal((4, 16)).astype(np.float32)
    query_segs /= np.linalg.norm(query_segs, axis=1, keepdims=True)

    numpy_out = similarity.top_k_neighbors(query_mean, query_segs, cat, k=3)
    faiss_out = similarity_faiss.top_k_neighbors_faiss(query_mean, query_segs, cat, k=3)

    # Same trackIds in the same order.
    assert [r["trackId"] for r in faiss_out] == [r["trackId"] for r in numpy_out]
    # Mean-pool similarity matches within float32 noise.
    for f, n in zip(faiss_out, numpy_out):
        assert f["meanPooledSimilarity"] == pytest.approx(n["meanPooledSimilarity"], abs=1e-5)
        assert f["maxSegmentSimilarity"] == pytest.approx(n["maxSegmentSimilarity"], abs=1e-5)


def test_faiss_returns_match_window_indices():
    """The matchQueryWindow / matchCatalogWindow fields are preserved + valid."""
    tracks, means, segs = _make_synthetic_catalog()
    cat = similarity.build_flat_catalog(tracks, means, segs)
    # t2 has 3 segments; argmax may pick any diagonal entry (all cosine 1.0),
    # so just verify the indices are valid and the max-segment cosine ~= 1.0.
    out = similarity_faiss.top_k_neighbors_faiss(means[2].copy(), segs["t2"].copy(), cat, k=1)
    assert "matchQueryWindow" in out[0]
    assert "matchCatalogWindow" in out[0]
    n_query_segs = segs["t2"].shape[0]
    n_catalog_segs = segs["t2"].shape[0]
    assert 0 <= out[0]["matchQueryWindow"] < n_query_segs
    assert 0 <= out[0]["matchCatalogWindow"] < n_catalog_segs
    assert out[0]["maxSegmentSimilarity"] == pytest.approx(1.0, abs=1e-5)


def test_faiss_clamps_k_to_catalog_size():
    tracks, means, segs = _make_synthetic_catalog()
    cat = similarity.build_flat_catalog(tracks, means, segs)
    out = similarity_faiss.top_k_neighbors_faiss(means[0].copy(), segs["t0"].copy(), cat, k=999)
    assert len(out) == 5


def test_faiss_empty_catalog_returns_empty():
    cat = similarity.FlatCatalog(track_ids=[], means=np.empty((0, 16), dtype=np.float32),
                                  segs_flat=np.empty((0, 16), dtype=np.float32), seg_ranges=[])
    rng = np.random.default_rng(0)
    q = rng.standard_normal(16).astype(np.float32)
    q /= np.linalg.norm(q)
    q_segs = q.reshape(1, -1).copy()
    assert similarity_faiss.top_k_neighbors_faiss(q, q_segs, cat, k=5) == []


def test_faiss_validates_dim_mismatch():
    tracks, means, segs = _make_synthetic_catalog()
    cat = similarity.build_flat_catalog(tracks, means, segs)
    rng = np.random.default_rng(0)
    bad_dim = rng.standard_normal(8).astype(np.float32)
    bad_dim /= np.linalg.norm(bad_dim)
    with pytest.raises(ValueError, match="does not match catalog dim"):
        similarity_faiss.top_k_neighbors_faiss(bad_dim, bad_dim.reshape(1, -1), cat, k=1)


def test_index_caching_reuses_built_index():
    """Two queries on the same catalog use one index, not two."""
    tracks, means, segs = _make_synthetic_catalog()
    cat = similarity.build_flat_catalog(tracks, means, segs)

    q = means[0].copy()
    q_segs = segs["t0"].copy()
    similarity_faiss.top_k_neighbors_faiss(q, q_segs, cat, k=1)
    cache_size_after_first = len(similarity_faiss._index_cache)
    similarity_faiss.top_k_neighbors_faiss(q, q_segs, cat, k=1)
    cache_size_after_second = len(similarity_faiss._index_cache)
    assert cache_size_after_first == 1
    assert cache_size_after_second == 1


def test_is_available_returns_true_in_test_env():
    """We installed faiss-cpu locally for the bench; tests inherit that."""
    assert similarity_faiss.is_available() is True
