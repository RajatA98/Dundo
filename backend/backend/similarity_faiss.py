"""FAISS-backed similarity — drop-in replacement for similarity.top_k_neighbors.

Selected by `SIMILARITY_BACKEND=faiss` env var. Default backend stays NumPy
(see similarity.py). This module proves the climb to the next rung is wired
without changing production behavior — see ADR-0007 for the trigger conditions.

Architecture:
    1. FAISS IndexFlatIP over the catalog's mean vectors. Same exact cosine as
       NumPy because catalog rows are L2-normalized; inner product = cosine.
       Sub-linear vs NumPy starts paying off around ~10k tracks (see
       SIMILARITY_BENCH.json).
    2. Index is built lazily on first query per `FlatCatalog` instance and
       cached by id(catalog.means). One-time cost amortizes across all
       subsequent queries.
    3. Cross-segment max similarity (the secondary metric) STILL uses the
       NumPy cross-matmul over the K selected tracks only — not the full
       catalog. This is the real production scaling pattern: FAISS selects
       the top-K cheaply, then per-track segment math runs on just those K.

Return shape is identical to similarity.top_k_neighbors:
    [{"trackId": str, "meanPooledSimilarity": float, "maxSegmentSimilarity": float,
      "matchQueryWindow": int, "matchCatalogWindow": int}, ...]

Switching backends should be invisible to api.py + the frontend.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .similarity import FlatCatalog

# id(catalog.means) → faiss index. Bare dict + lock is sufficient — catalogs
# are constructed once at startup and we don't need eviction.
_index_cache: dict[int, Any] = {}
_index_lock = threading.Lock()


def _get_or_build_index(catalog: FlatCatalog):
    """Lazily build the FAISS IndexFlatIP for this catalog. Thread-safe."""
    import faiss  # deferred import — faiss is only required when backend=faiss.

    key = id(catalog.means)
    with _index_lock:
        index = _index_cache.get(key)
        if index is not None:
            return index
        # IndexFlatIP: exact inner product. L2-normalized vectors → cosine.
        # FAISS expects float32 contiguous arrays; api.py guarantees both.
        index = faiss.IndexFlatIP(catalog.means.shape[1])
        index.add(catalog.means)
        _index_cache[key] = index
        return index


def top_k_neighbors_faiss(
    query_mean: np.ndarray,
    query_segs: np.ndarray,
    catalog: FlatCatalog,
    k: int = 5,
) -> list[dict]:
    """FAISS Flat top-K + NumPy cross-segment for the K winners.

    Same contract as similarity.top_k_neighbors. See that module for arg /
    return docs. This implementation is exact (FAISS IndexFlatIP), not
    approximate — same numbers as NumPy, just faster at scale.
    """
    n = len(catalog.track_ids)
    if n == 0:
        return []

    query_mean_arr = np.asarray(query_mean, dtype=np.float32)
    query_segs_arr = np.asarray(query_segs, dtype=np.float32)
    if query_mean_arr.ndim != 1:
        raise ValueError(f"query_mean must be 1-D, got shape {query_mean_arr.shape}")
    if query_segs_arr.ndim != 2 or query_segs_arr.shape[0] == 0:
        raise ValueError(f"query_segs must be non-empty 2-D, got shape {query_segs_arr.shape}")
    if query_mean_arr.shape[0] != catalog.means.shape[1]:
        raise ValueError(
            f"query dim {query_mean_arr.shape[0]} does not match catalog dim {catalog.means.shape[1]}"
        )
    if query_segs_arr.shape[1] != catalog.segs_flat.shape[1]:
        raise ValueError(
            f"query segment dim {query_segs_arr.shape[1]} does not match catalog dim {catalog.segs_flat.shape[1]}"
        )

    k = max(1, min(int(k), n))

    # FAISS top-K via inner product. The catalog is L2-normalized so this is
    # exact cosine. `D` shape: (1, k); `I` shape: (1, k).
    index = _get_or_build_index(catalog)
    D, I = index.search(query_mean_arr.reshape(1, -1), k)
    top_indices = I[0]
    top_mean_sims = D[0]

    # Cross-segment max similarity for the K winners only. This is the real
    # production scaling pattern — at 1M tracks we'd still only run this
    # Q × (k × 3) matmul, not Q × (N × 3).
    results: list[dict] = []
    for rank in range(k):
        i = int(top_indices[rank])
        mean_sim = float(top_mean_sims[rank])
        start, end = catalog.seg_ranges[i]
        sub = query_segs_arr @ catalog.segs_flat[start:end].T  # (Q, segs_for_i)
        flat_idx = int(sub.argmax())
        qi, cj = np.unravel_index(flat_idx, sub.shape)
        results.append({
            "trackId": catalog.track_ids[i],
            "meanPooledSimilarity": mean_sim,
            "maxSegmentSimilarity": float(sub[qi, cj]),
            "matchQueryWindow": int(qi),
            "matchCatalogWindow": int(cj),
        })

    return results


def is_available() -> bool:
    """True if faiss can be imported. Used by api.py to fail loud if the
    SIMILARITY_BACKEND=faiss flag is set but the dep is missing."""
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def clear_index_cache() -> None:
    """Wipe the index cache. Used by tests to isolate per-test catalogs."""
    with _index_lock:
        _index_cache.clear()
