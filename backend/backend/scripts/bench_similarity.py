"""Benchmark NumPy vs FAISS Flat vs FAISS HNSW on synthetic catalogs.

ADR-0007 (RAG scaling) needs measured numbers, not just claims. This script
synthesizes L2-normalized random catalogs at 1k / 10k / 100k / 1M tracks
and measures top-K query latency for three backends:

  1. NumPy: `means @ query.T` then argpartition + sort. Current production.
  2. FAISS IndexFlatIP: exact cosine via inner product (catalog is L2-
     normalized, so inner product = cosine). SIMD-accelerated; same numbers.
  3. FAISS IndexHNSWFlat: approximate nearest neighbor. Sublinear search;
     recall@1 ~0.95 at default ef_search.

For each backend, report query p50/p95/p99 latency across 200 queries.
Output is a markdown table that can be pasted directly into ADR-0007.

Run:
    python -m backend.scripts.bench_similarity
    python -m backend.scripts.bench_similarity --sizes 1000 10000 100000

Optional --json writes the result to disk for the ADR exhibit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

EMBED_DIM = 512  # matches MuQ-MuLan AUDIO_ENCODER_EMBED_DIM
TOP_K = 5
N_QUERIES = 200
RNG_SEED = 42


def make_catalog(n: int, dim: int = EMBED_DIM, seed: int = RNG_SEED) -> np.ndarray:
    """Random L2-normalized float32 catalog matrix (N, dim)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def make_queries(n: int, dim: int = EMBED_DIM, seed: int = RNG_SEED + 1) -> np.ndarray:
    """Random L2-normalized float32 query batch (n, dim)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def numpy_search(catalog: np.ndarray, queries: np.ndarray, k: int) -> list[float]:
    """Per-query latency in ms for NumPy means @ query + argpartition."""
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        sims = catalog @ q  # (N,)
        # Match the api.py pattern: argpartition + sort the top-K.
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            order = part[np.argsort(-sims[part])]
        _ = order  # ensure side effect
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def faiss_flat_search(catalog: np.ndarray, queries: np.ndarray, k: int) -> list[float]:
    """Per-query latency for FAISS IndexFlatIP (exact inner product = cosine)."""
    import faiss

    index = faiss.IndexFlatIP(catalog.shape[1])
    index.add(catalog)
    latencies: list[float] = []
    for q in queries:
        q_batch = q.reshape(1, -1)
        t0 = time.perf_counter()
        _D, _I = index.search(q_batch, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def faiss_hnsw_search(catalog: np.ndarray, queries: np.ndarray, k: int) -> list[float]:
    """Per-query latency for FAISS IndexHNSWFlat (approximate, sublinear)."""
    import faiss

    # M=32 is the FAISS docs' default and a good speed/recall tradeoff.
    index = faiss.IndexHNSWFlat(catalog.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(catalog)
    latencies: list[float] = []
    for q in queries:
        q_batch = q.reshape(1, -1)
        t0 = time.perf_counter()
        _D, _I = index.search(q_batch, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    rank = p * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + frac * (s[hi] - s[lo])


def bench_one(
    backend_name: str,
    fn: Callable[[np.ndarray, np.ndarray, int], list[float]],
    catalog: np.ndarray,
    queries: np.ndarray,
) -> dict:
    """Run one backend × catalog-size and return summary stats."""
    # Warm-up: 5 queries to amortize one-time costs (FAISS index build).
    warmup_queries = queries[:5]
    fn(catalog, warmup_queries, TOP_K)

    # Real run.
    samples = fn(catalog, queries, TOP_K)
    return {
        "backend": backend_name,
        "n": int(catalog.shape[0]),
        "queries": len(samples),
        "p50_ms": round(percentile(samples, 0.50), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "p99_ms": round(percentile(samples, 0.99), 3),
        "mean_ms": round(float(np.mean(samples)), 3),
    }


def run(sizes: list[int], out_json: Path | None) -> dict:
    queries = make_queries(N_QUERIES)
    rows: list[dict] = []
    for n in sizes:
        print(f"[bench] building catalog of {n} tracks…", flush=True)
        catalog = make_catalog(n)
        for backend_name, fn in [
            ("numpy", numpy_search),
            ("faiss_flat", faiss_flat_search),
            ("faiss_hnsw", faiss_hnsw_search),
        ]:
            print(f"[bench]   {backend_name}…", end=" ", flush=True)
            row = bench_one(backend_name, fn, catalog, queries)
            rows.append(row)
            print(f"p50={row['p50_ms']}ms p95={row['p95_ms']}ms p99={row['p99_ms']}ms")

    summary = {
        "embed_dim": EMBED_DIM,
        "top_k": TOP_K,
        "n_queries": N_QUERIES,
        "rows": rows,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[bench] wrote {out_json}")
    return summary


def print_markdown_table(summary: dict) -> None:
    """Render the ADR-0007 exhibit table."""
    rows = summary["rows"]
    sizes = sorted({r["n"] for r in rows})
    backends = ["numpy", "faiss_flat", "faiss_hnsw"]

    print()
    print(f"# Backend × catalog size — top-{summary['top_k']} query p50 latency (ms)")
    print()
    header = "| Catalog | " + " | ".join(backends) + " |"
    sep = "|---:|" + "|".join([":---:" for _ in backends]) + "|"
    print(header)
    print(sep)
    for n in sizes:
        cells = [f"{n:,}"]
        for b in backends:
            match = next((r for r in rows if r["n"] == n and r["backend"] == b), None)
            cells.append(f"{match['p50_ms']:.2f}" if match else "-")
        print("| " + " | ".join(cells) + " |")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[155, 1_000, 10_000, 100_000],
        help="Catalog sizes to benchmark (default: 155 1000 10000 100000).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON (e.g. for ADR-0007 exhibit).",
    )
    args = parser.parse_args()

    summary = run(args.sizes, args.json)
    print_markdown_table(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
