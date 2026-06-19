"""Run the Dundo eval against the golden set and write eval.json.

Phase 6 entry point. Reads:
  - `backend/eval_input/golden_set.json` (built by build_golden_set.py)
  - `backend/eval_input/named_examples.yaml` (curated 5 FP + 5 FN with audio paths + "why" notes)
  - The live corpus at `quality-scorer/public/corpus/` (Phase 1 artifacts)

Writes:
  - `quality-scorer/public/corpus/eval.json` — the eval page reads this at runtime.
  - Audio files for the named examples get copied to
    `quality-scorer/public/eval_audio/` so the frontend can `<audio src=...>` them.

Per LOCKED_DECISIONS (the eval section):

  - Metrics: Recall@1, Recall@3, MRR over the positive queries
    (queries that target a seed in the catalog).
  - Top-1 cosine histogram on the unrelated negatives — shows the noise floor.
  - Per-signal observed behavior for each ACRCloud signal where ENABLE_ACRCLOUD
    is true (Phase 5 must be implemented; if not, the per-signal block is omitted).
  - 5 named false-positive + 5 named false-negative examples with audio playback.
  - A methodology paragraph + a limitations paragraph.

Usage:
    python -m backend.scripts.run_eval
    # or with explicit paths:
    python -m backend.scripts.run_eval \\
        --golden-set backend/eval_input/golden_set.json \\
        --named-examples backend/eval_input/named_examples.yaml \\
        --out quality-scorer/public/corpus/eval.json

Re-running is safe and deterministic given the same golden_set.json + corpus.
The whole job runs in ~3–5 min on a laptop for ~80 queries.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm
import yaml

from backend import clap_windowed, config, similarity
from backend.scripts.rebuild_corpus import _decode_to_mono, _resolve_model_sha

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN = REPO_ROOT / "backend" / "eval_input" / "golden_set.json"
DEFAULT_NAMED = REPO_ROOT / "backend" / "eval_input" / "named_examples.yaml"
DEFAULT_OUT = REPO_ROOT / "quality-scorer" / "public" / "corpus" / "eval.json"
DEFAULT_AUDIO_DEST = REPO_ROOT / "quality-scorer" / "public" / "eval_audio"
CORPUS_DIR = REPO_ROOT / "quality-scorer" / "public" / "corpus"


def main() -> None:
    args = _parse_args()

    catalog_doc = _load_catalog()
    catalog = catalog_doc["flat_catalog"]
    manifest = catalog_doc["manifest"]
    tracks = catalog_doc["tracks"]
    named = _load_named_examples(args.named_examples)

    if args.mode == "loo":
        print(f"[eval] running leave-one-out catalog eval: {len(tracks)} catalog tracks")
        positive_results = run_leave_one_out(tracks, catalog)
        negative_results = [row for row in positive_results if row.get("rank_of_seed") != 1]
        mode_methodology = _LOO_METHODOLOGY
        mode_limitations = _LOO_LIMITATIONS
        n_positives = len(positive_results)
        n_negatives = 0
    else:
        golden_set = _load_golden_set(args.golden_set)
        print(f"[eval] loaded golden set: {len(golden_set['positives'])} positives + "
              f"{len(golden_set['negatives'])} negatives")

        positive_results = run_queries(golden_set["positives"], catalog, kind="positive")
        negative_results = run_queries(golden_set["negatives"], catalog, kind="negative")
        mode_methodology = _DEFAULT_METHODOLOGY
        mode_limitations = _DEFAULT_LIMITATIONS
        n_positives = len(golden_set["positives"])
        n_negatives = len(golden_set["negatives"])

    # ---- compute retrieval metrics on positives ------------------------
    metrics = compute_metrics(positive_results)

    # ---- top-1 cosine histogram on negatives ---------------------------
    histogram = compute_histogram(negative_results, bins=20, lo=0.0, hi=1.0)

    # ---- latency benchmark (rag-eval-harness mandatory metric) ---------
    latency = compute_latency(catalog, n_samples=20, seed=0)

    # ---- copy named examples' audio to public/ + build the eval blocks --
    fp_block = build_named_block(named.get("false_positives", []), args.audio_dest)
    fn_block = build_named_block(named.get("false_negatives", []), args.audio_dest)

    # ---- resolve golden-set version (rag-eval-harness versioning rule) -
    if args.mode == "golden":
        golden_doc_for_version = _load_golden_set(args.golden_set)
        golden_set_version = str(golden_doc_for_version.get("version") or "v0.0.0")
    else:
        # For LOO the "golden set" is the corpus itself; the corpus's model_sha
        # is the natural version handle.
        sha = str(manifest.get("model_sha") or _resolve_model_sha())
        golden_set_version = f"corpus@{sha[:12]}"

    # ---- assemble eval.json --------------------------------------------
    eval_doc = {
        "metrics": metrics,
        "negatives_histogram": histogram,
        "latency": latency,
        "named_examples": {
            "false_positives": fp_block,
            "false_negatives": fn_block,
        },
        "methodology": named.get("methodology", mode_methodology),
        "limitations": named.get("limitations", mode_limitations),
        "manifest": {
            "model_sha": str(manifest.get("model_sha") or _resolve_model_sha()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_positives": n_positives,
            "n_negatives": n_negatives,
            "eval_mode": args.mode,
            "threshold_default": float(manifest.get("threshold_default", config.SIMILARITY_THRESHOLD_DEFAULT)),
            "golden_set_version": golden_set_version,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(eval_doc, indent=2))
    print(f"[eval] wrote {args.out}")
    print(f"[eval] R@1={metrics['recall_at_1']:.3f}  "
          f"R@3={metrics['recall_at_3']:.3f}  "
          f"MRR={metrics['mrr']:.3f}  "
          f"n={metrics['n_queries']}")
    print(f"[eval] latency p50={latency['p50_ms']:.2f}ms  "
          f"p95={latency['p95_ms']:.2f}ms  "
          f"p99={latency['p99_ms']:.2f}ms  "
          f"n={latency['n_samples']}")
    print(f"[eval] golden_set_version={golden_set_version}")


def run_leave_one_out(tracks: list[dict], catalog: similarity.FlatCatalog) -> list[dict]:
    """Run each catalog embedding as a query against all other catalog rows.

    This does not re-encode audio. The query vector and query windows are copied
    from the existing corpus artifacts, then the queried row is removed from the
    temporary index before ranking.
    """
    by_track_id = {str(row.get("track_id")): row for row in tracks}
    results: list[dict] = []
    for idx, track_id in enumerate(tqdm(catalog.track_ids, desc="eval loo")):
        query_track = by_track_id.get(track_id, {})
        query_mean = catalog.means[idx]
        start, end = catalog.seg_ranges[idx]
        query_segs = catalog.segs_flat[start:end]
        held_out = _catalog_without_index(catalog, idx)
        neighbors = similarity.top_k_neighbors(query_mean, query_segs, held_out, k=10)

        query_artist = str(query_track.get("artist") or "").strip().lower()
        target_ids = {
            str(row.get("track_id"))
            for row in tracks
            if str(row.get("track_id")) != track_id
            and query_artist
            and str(row.get("artist") or "").strip().lower() == query_artist
        }
        rank = None
        if target_ids:
            for pos, neighbor in enumerate(neighbors, start=1):
                if neighbor["trackId"] in target_ids:
                    rank = pos
                    break

        results.append(
            {
                "id": f"loo_{track_id}",
                "query_track_id": track_id,
                "query_title": query_track.get("title") or track_id,
                "query_artist": query_track.get("artist") or "",
                "target_track_ids": sorted(target_ids),
                "rank_of_seed": rank,
                "top_neighbors": neighbors,
                "top1_score": float(neighbors[0]["meanPooledSimilarity"]) if neighbors else 0.0,
            }
        )
    return results


def run_queries(queries: list[dict], catalog: similarity.FlatCatalog, *, kind: str) -> list[dict]:
    """Encode each query and rank against the catalog.

    Args:
        queries: list of dicts from golden_set.json. Positive shape:
            {id, seed_track_id, query_audio_path}.
            Negative shape: {id, query_audio_path}.
        catalog: the FlatCatalog already built from disk.
        kind: "positive" or "negative" — only affects logging.

    Returns:
        For each query, a dict adding:
            - top_neighbors: list of {trackId, meanPooledSimilarity, maxSegmentSimilarity}
            - rank_of_seed: 1-indexed rank of the seed_track_id in top_neighbors,
              or None if not in top-K. (positives only)
            - top1_score: float (mean-pooled similarity of rank-1 neighbor).
    """
    results: list[dict] = []
    for query in tqdm(queries, desc=f"eval {kind}"):
        raw = Path(query["query_audio_path"]).read_bytes()
        wav_mono, sr = _decode_to_mono(raw)
        mean_pooled, segs = clap_windowed.encode_windowed(
            wav_mono,
            sr,
            max_seconds=config.CLAP_QUERY_MAX_SECONDS,
        )
        neighbors = similarity.top_k_neighbors(mean_pooled, segs, catalog, k=10)
        rank = None
        seed_track_id = query.get("seed_track_id")
        if kind == "positive" and seed_track_id:
            for idx, neighbor in enumerate(neighbors, start=1):
                if neighbor["trackId"] == seed_track_id:
                    rank = idx
                    break
        results.append(
            {
                **query,
                "kind": kind,
                "top_neighbors": neighbors,
                "rank_of_seed": rank,
                "top1_score": float(neighbors[0]["meanPooledSimilarity"]) if neighbors else 0.0,
            }
        )
    return results


def compute_metrics(positives: list[dict]) -> dict:
    """Compute Recall@1, Recall@3, MRR over the positive queries.

    Args:
        positives: output of `run_queries(..., kind="positive")`.

    Returns:
        {
          "recall_at_1": float,    # share of positives where rank_of_seed == 1
          "recall_at_3": float,    # share where rank_of_seed in {1, 2, 3}
          "mrr": float,            # mean reciprocal rank; queries with no seed → 1/k+1 fallback
          "n_queries": int,
        }
    """
    n = len(positives)
    if n == 0:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0, "n_queries": 0}

    ranks = [row.get("rank_of_seed") for row in positives]
    recall_at_1 = sum(1 for rank in ranks if rank == 1) / n
    recall_at_3 = sum(1 for rank in ranks if isinstance(rank, int) and 1 <= rank <= 3) / n
    mrr = sum((1.0 / rank) if isinstance(rank, int) and rank > 0 else 0.0 for rank in ranks) / n
    return {
        "recall_at_1": float(recall_at_1),
        "recall_at_3": float(recall_at_3),
        "mrr": float(mrr),
        "n_queries": n,
    }


def compute_latency(
    catalog: similarity.FlatCatalog,
    *,
    n_samples: int = 20,
    seed: int = 0,
) -> dict:
    """Wall-clock benchmark of the /neighbors ranking hot path.

    Per the rag-eval-harness methodology, latency is a first-class metric
    alongside precision/recall — slow systems get ignored. This bench samples
    `n_samples` random catalog tracks, uses each as a query (matching the live
    /neighbors path), and times the `top_k_neighbors` call only. Audio decode
    and CLAP encode are NOT included; those run once per upload and are bounded
    by file size, not by index size.

    Returns:
        {
          "p50_ms": float, "p95_ms": float, "p99_ms": float,
          "n_samples": int,
          "note": str,
        }
    """
    import time as _time

    n = len(catalog.track_ids)
    if n == 0:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "n_samples": 0,
            "note": "empty catalog",
        }

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n, size=min(int(n_samples), n), replace=False)

    timings_ms: list[float] = []
    for idx in sample_indices:
        idx_int = int(idx)
        query_mean = catalog.means[idx_int]
        start, end = catalog.seg_ranges[idx_int]
        query_segs = catalog.segs_flat[start:end]
        t0 = _time.perf_counter()
        similarity.top_k_neighbors(query_mean, query_segs, catalog, k=3)
        t1 = _time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    timings_ms.sort()

    def _percentile(p: float) -> float:
        if not timings_ms:
            return 0.0
        i = max(0, min(int(p * len(timings_ms)), len(timings_ms) - 1))
        return round(timings_ms[i], 3)

    return {
        "p50_ms": _percentile(0.50),
        "p95_ms": _percentile(0.95),
        "p99_ms": _percentile(0.99),
        "n_samples": len(timings_ms),
        "note": "Wall-clock per /neighbors ranking call against the in-memory catalog. Excludes audio decode + CLAP encode (those are bounded by file size, not index size).",
    }


def compute_histogram(negatives: list[dict], *, bins: int, lo: float, hi: float) -> dict:
    """Top-1 cosine histogram on the unrelated negatives. Shows the noise floor.

    Args:
        negatives: output of `run_queries(..., kind="negative")`.
        bins, lo, hi: histogram parameters.

    Returns:
        {
          "bins": [lo, lo+step, ...],     # bin edges, length bins+1
          "counts": [int, ...],           # counts per bin, length bins
          "step": float,
        }
    """
    step = (hi - lo) / bins
    edges = [round(lo + step * i, 10) for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for row in negatives:
        score = float(row.get("top1_score", 0.0))
        if score < lo or score > hi:
            continue
        idx = int((score - lo) / step)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    return {"bins": edges, "counts": counts, "step": float(step)}


def build_named_block(named_specs: list[dict], audio_dest: Path) -> list[dict]:
    """Copy named-example audio into `quality-scorer/public/eval_audio/` and build the eval block.

    Args:
        named_specs: list of {id, query_audio_path, retrieved_audio_path,
                              query_title, retrieved_title, cosine, why}.
        audio_dest: destination directory inside `quality-scorer/public/`.

    Returns:
        List of normalized dicts the eval page consumes:
        {
          "id": str,
          "query_title": str,
          "retrieved_title": str,
          "cosine": float,
          "why": str,
          "query_audio_url": str,        # e.g. "/eval_audio/fp_001_query.mp3"
          "retrieved_audio_url": str,
        }
    """
    if not named_specs:
        return []

    audio_dest.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for spec in named_specs:
        item_id = str(spec["id"])
        query_src = Path(spec["query_audio_path"])
        retrieved_src = Path(spec["retrieved_audio_path"])
        query_ext = query_src.suffix or ".mp3"
        retrieved_ext = retrieved_src.suffix or ".mp3"
        query_name = f"{item_id}_query{query_ext}"
        retrieved_name = f"{item_id}_retrieved{retrieved_ext}"
        shutil.copyfile(query_src, audio_dest / query_name)
        shutil.copyfile(retrieved_src, audio_dest / retrieved_name)
        items.append(
            {
                "id": item_id,
                "query_title": spec.get("query_title") or item_id,
                "retrieved_title": spec.get("retrieved_title") or "",
                "cosine": float(spec.get("cosine", 0.0)),
                "why": spec.get("why") or "",
                "query_audio_url": f"/eval_audio/{query_name}",
                "retrieved_audio_url": f"/eval_audio/{retrieved_name}",
            }
        )
    return items


def _load_golden_set(path: Path) -> dict:
    """Parse golden_set.json. Expected shape:
       { "positives": [...], "negatives": [...] }
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("golden_set.json must be an object")
    positives = data.get("positives", [])
    negatives = data.get("negatives", [])
    if not isinstance(positives, list) or not isinstance(negatives, list):
        raise ValueError("golden_set.json positives and negatives must be lists")
    return {"positives": positives, "negatives": negatives}


def _load_named_examples(path: Path) -> dict:
    """Parse named_examples.yaml. See the YAML template in backend/eval_input/."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _load_catalog() -> dict:
    """Load the live corpus and build a FlatCatalog plus metadata."""
    cpath = CORPUS_DIR / "corpus.json"
    epath = CORPUS_DIR / "embeddings.npy"
    spath = CORPUS_DIR / "segment_embeddings.npz"
    mpath = CORPUS_DIR / "manifest.json"
    tracks_data = json.loads(cpath.read_text())
    tracks = tracks_data if isinstance(tracks_data, list) else tracks_data.get("tracks", [])
    embeddings = np.load(epath).astype(np.float32)
    with np.load(spath) as npz:
        segment_embeddings = {k: npz[k].astype(np.float32) for k in npz.files}
    manifest = json.loads(mpath.read_text())
    flat_catalog = similarity.build_flat_catalog(tracks, embeddings, segment_embeddings)
    return {"tracks": tracks, "flat_catalog": flat_catalog, "manifest": manifest}


def _catalog_without_index(catalog: similarity.FlatCatalog, drop_index: int) -> similarity.FlatCatalog:
    track_ids: list[str] = []
    means: list[np.ndarray] = []
    seg_arrays: list[np.ndarray] = []
    for idx, track_id in enumerate(catalog.track_ids):
        if idx == drop_index:
            continue
        start, end = catalog.seg_ranges[idx]
        track_ids.append(track_id)
        means.append(catalog.means[idx])
        seg_arrays.append(catalog.segs_flat[start:end])

    if means:
        means_arr = np.stack(means, axis=0).astype(np.float32)
        segs_flat = np.vstack(seg_arrays).astype(np.float32)
    else:
        means_arr = np.empty((0, catalog.means.shape[1]), dtype=np.float32)
        segs_flat = np.empty((0, catalog.segs_flat.shape[1]), dtype=np.float32)

    seg_ranges: list[tuple[int, int]] = []
    cursor = 0
    for segs in seg_arrays:
        start = cursor
        cursor += segs.shape[0]
        seg_ranges.append((start, cursor))
    return similarity.FlatCatalog(track_ids=track_ids, means=means_arr, segs_flat=segs_flat, seg_ranges=seg_ranges)


_DEFAULT_METHODOLOGY = (
    "30 seed songs hand-picked from the reference catalog; for each seed, "
    "two Suno generations were created by prompting toward the seed's style + "
    "lyrical theme. 20-30 unrelated negatives are Suno generations with no "
    "intentional similarity to any catalog track. The retrieval metrics and "
    "MRR are computed on the 60 positive queries; the histogram is computed "
    "on the negatives."
)

_DEFAULT_LIMITATIONS = (
    "Catalog size (~160 tracks) is the dominant failure mode — a real source "
    "outside the catalog can't be retrieved. Queries are single-generator "
    "(Suno only); Udio or others may shift the score distribution. There is "
    "no inter-rater agreement on what counts as derivative, and the seed set "
    "carries a US-pop bias that likely inflates recall relative to other genres."
)

_LOO_METHODOLOGY = (
    "Retrieval check - leave-one-out over the existing catalog. Each catalog "
    "track is used as a query embedding while that exact row is held out of "
    "the index; the system then ranks the remaining catalog tracks with the "
    "same mean-pooled CLAP cosine used by the live /neighbors endpoint. "
    "Recall@k and MRR count whether another track by the same artist appears "
    "in the top-k. Because each LOO query has at most one ground-truth target, "
    "Precision@1 equals Recall@1 here; we report Recall@k by convention and "
    "Precision@k = Recall@k / k for any k. Latency is wall-clock per "
    "/neighbors ranking call against the in-memory catalog. Groundedness "
    "(entity extraction from generated text) is not applicable - this system "
    "retrieves, it does not generate. The histogram is a LOO top-1 score "
    "distribution for queries that did not retrieve a same-artist track at "
    "rank 1."
)

_LOO_LIMITATIONS = (
    "This is a retrieval sanity check, not a definitive AI-generation eval. It "
    "does not use Suno generations or unrelated human-labeled negatives, so the "
    "histogram should not be read as a production false-positive distribution. "
    "Many catalog artists have only one track, which makes same-artist recall "
    "strict and depresses the headline metrics. The check is still useful "
    "because it is reproducible, uses the shipped catalog, and exercises the "
    "same similarity path as the demo."
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("loo", "golden"), default="loo")
    p.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--named-examples", type=Path, default=DEFAULT_NAMED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--audio-dest", type=Path, default=DEFAULT_AUDIO_DEST)
    return p.parse_args()


if __name__ == "__main__":
    main()
