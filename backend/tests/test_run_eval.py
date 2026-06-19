"""Phase 6 — eval pipeline unit tests.

Fast tests against synthetic per-query results — no CLAP, no corpus, no audio.
The slow end-to-end smoke test (encoding real audio through the live catalog) is
opt-in via the `slow` marker and depends on a populated corpus, so it's gated.

The tests codify the contracts the `EvaluationPage.jsx` (Claude, Phase 6 final)
will read against. The wire shape is the JSON these functions produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# compute_metrics — Recall@1, Recall@3, MRR on positive queries
# ---------------------------------------------------------------------------


def test_compute_metrics_all_seeds_at_rank_one():
    from backend.scripts.run_eval import compute_metrics

    positives = [
        {"id": "p1", "rank_of_seed": 1, "top1_score": 0.91},
        {"id": "p2", "rank_of_seed": 1, "top1_score": 0.88},
        {"id": "p3", "rank_of_seed": 1, "top1_score": 0.85},
    ]
    m = compute_metrics(positives)
    assert m["recall_at_1"] == pytest.approx(1.0)
    assert m["recall_at_3"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx(1.0)
    assert m["n_queries"] == 3


def test_compute_metrics_mixed_ranks():
    from backend.scripts.run_eval import compute_metrics

    positives = [
        {"id": "p1", "rank_of_seed": 1, "top1_score": 0.91},  # 1/1
        {"id": "p2", "rank_of_seed": 3, "top1_score": 0.55},  # 1/3
        {"id": "p3", "rank_of_seed": None, "top1_score": 0.45},  # not in top-K → 0
        {"id": "p4", "rank_of_seed": 2, "top1_score": 0.72},  # 1/2
    ]
    m = compute_metrics(positives)
    assert m["recall_at_1"] == pytest.approx(0.25)            # 1 of 4
    assert m["recall_at_3"] == pytest.approx(0.75)            # 3 of 4 (ranks 1, 2, 3)
    # MRR = (1 + 1/3 + 0 + 1/2) / 4
    assert m["mrr"] == pytest.approx((1 + 1/3 + 0 + 1/2) / 4, abs=1e-6)
    assert m["n_queries"] == 4


def test_compute_metrics_zero_positives():
    """Empty input must not crash; returns all zeros with n_queries=0."""
    from backend.scripts.run_eval import compute_metrics

    m = compute_metrics([])
    assert m["recall_at_1"] == 0.0
    assert m["recall_at_3"] == 0.0
    assert m["mrr"] == 0.0
    assert m["n_queries"] == 0


# ---------------------------------------------------------------------------
# compute_histogram — top-1 cosine distribution on negatives
# ---------------------------------------------------------------------------


def test_compute_histogram_bin_counts():
    from backend.scripts.run_eval import compute_histogram

    negatives = [
        {"id": "n1", "top1_score": 0.05},
        {"id": "n2", "top1_score": 0.14},
        {"id": "n3", "top1_score": 0.42},
        {"id": "n4", "top1_score": 0.43},
        {"id": "n5", "top1_score": 0.85},
    ]
    out = compute_histogram(negatives, bins=10, lo=0.0, hi=1.0)
    assert out["step"] == pytest.approx(0.1)
    assert len(out["bins"]) == 11   # bin edges
    assert len(out["counts"]) == 10
    # Bin 0: [0.0, 0.1) → 1 (just 0.05)
    # Bin 1: [0.1, 0.2) → 1 (just 0.14)
    # Bin 4: [0.4, 0.5) → 2 (0.42, 0.43)
    # Bin 8: [0.8, 0.9) → 1 (0.85)
    assert out["counts"][0] == 1
    assert out["counts"][1] == 1
    assert out["counts"][4] == 2
    assert out["counts"][8] == 1
    # Total preserved
    assert sum(out["counts"]) == len(negatives)


def test_compute_histogram_empty_negatives():
    from backend.scripts.run_eval import compute_histogram

    out = compute_histogram([], bins=20, lo=0.0, hi=1.0)
    assert sum(out["counts"]) == 0
    assert len(out["counts"]) == 20


# ---------------------------------------------------------------------------
# compute_latency — wall-clock benchmark of the /neighbors ranking hot path
# ---------------------------------------------------------------------------


def test_compute_latency_returns_p50_p95_p99_with_sample_count():
    import numpy as np
    from backend import similarity
    from backend.scripts.run_eval import compute_latency

    rng = np.random.default_rng(0)
    n_tracks = 30
    embed_dim = 16
    tracks = [{"track_id": f"t{i}", "title": f"Track {i}", "artist": "A"} for i in range(n_tracks)]
    means = rng.standard_normal((n_tracks, embed_dim)).astype("float32")
    means /= __import__("numpy").linalg.norm(means, axis=1, keepdims=True)
    segs = {f"t{i}": means[i:i+1].copy() for i in range(n_tracks)}
    catalog = similarity.build_flat_catalog(tracks, means, segs)

    out = compute_latency(catalog, n_samples=10, seed=0)

    assert {"p50_ms", "p95_ms", "p99_ms", "n_samples", "note"} <= set(out.keys())
    assert out["n_samples"] == 10
    assert out["p50_ms"] >= 0
    assert out["p95_ms"] >= out["p50_ms"]
    assert out["p99_ms"] >= out["p95_ms"]
    assert "ranking" in out["note"].lower()


def test_compute_latency_handles_empty_catalog():
    from backend import similarity
    from backend.scripts.run_eval import compute_latency
    import numpy as np

    empty = similarity.FlatCatalog(
        track_ids=[],
        means=np.empty((0, 16), dtype=np.float32),
        segs_flat=np.empty((0, 16), dtype=np.float32),
        seg_ranges=[],
    )
    out = compute_latency(empty, n_samples=5)
    assert out["n_samples"] == 0
    assert out["p50_ms"] == 0.0


# ---------------------------------------------------------------------------
# build_named_block — copies audio into public/eval_audio/ + structures the JSON
# ---------------------------------------------------------------------------


def test_build_named_block_copies_audio_and_emits_urls(tmp_path):
    from backend.scripts.run_eval import build_named_block

    # Create fake source mp3 stubs.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    q_file = src_dir / "q.mp3"
    r_file = src_dir / "r.mp3"
    q_file.write_bytes(b"\x00\x00")
    r_file.write_bytes(b"\x00\x00")

    audio_dest = tmp_path / "eval_audio"
    specs = [
        {
            "id": "fp_001",
            "query_audio_path": str(q_file),
            "retrieved_audio_path": str(r_file),
            "query_title": "Suno Pop Generation",
            "retrieved_title": "Blinding Lights — The Weeknd",
            "cosine": 0.72,
            "why": "Shared synthwave production palette.",
        },
    ]
    out = build_named_block(specs, audio_dest)

    assert len(out) == 1
    item = out[0]
    assert item["id"] == "fp_001"
    assert item["query_title"] == "Suno Pop Generation"
    assert item["retrieved_title"] == "Blinding Lights — The Weeknd"
    assert item["cosine"] == 0.72
    assert item["why"].startswith("Shared synthwave")

    # URLs must be public-relative paths under /eval_audio/.
    assert item["query_audio_url"].startswith("/eval_audio/")
    assert item["retrieved_audio_url"].startswith("/eval_audio/")

    # Files actually copied.
    copied_q = audio_dest / Path(item["query_audio_url"]).name
    copied_r = audio_dest / Path(item["retrieved_audio_url"]).name
    assert copied_q.exists()
    assert copied_r.exists()


def test_build_named_block_empty_input():
    from backend.scripts.run_eval import build_named_block

    out = build_named_block([], Path("/tmp/should_not_be_created"))
    assert out == []


# ---------------------------------------------------------------------------
# build_golden_set.validate — user's YAML expands into structured JSON rows
# ---------------------------------------------------------------------------


def test_build_golden_set_validate_expands_positive_audio_paths(tmp_path):
    from backend.scripts.build_golden_set import validate

    pos_a = tmp_path / "pos_a.mp3"
    pos_b = tmp_path / "pos_b.mp3"
    neg = tmp_path / "neg.mp3"
    for path in (pos_a, pos_b, neg):
        path.write_bytes(b"\x00\x00")

    raw = {
        "positives": [
            {
                "seed_track_id": "tier1:itunes:1499378034",
                "seed_title": "Blinding Lights",
                "suno_audio_paths": [str(pos_a), str(pos_b)],
                "prompt_used": "retro synth pop",
            }
        ],
        "negatives": [
            {
                "id": "neg_001",
                "audio_path": str(neg),
                "prompt_used": "ambient drone",
            }
        ],
    }

    out = validate(raw, base_dir=tmp_path)

    assert [p["id"] for p in out["positives"]] == [
        "pos_1499378034_v1",
        "pos_1499378034_v2",
    ]
    assert out["positives"][0]["seed_track_id"] == "tier1:itunes:1499378034"
    assert out["positives"][0]["seed_title"] == "Blinding Lights"
    assert out["positives"][0]["prompt_used"] == "retro synth pop"
    assert Path(out["positives"][0]["query_audio_path"]).is_absolute()
    assert out["negatives"][0]["id"] == "neg_001"
    assert Path(out["negatives"][0]["query_audio_path"]).is_absolute()


def test_build_golden_set_validate_rejects_missing_audio(tmp_path):
    from backend.scripts.build_golden_set import validate

    raw = {
        "positives": [
            {
                "seed_track_id": "tier1:itunes:1499378034",
                "suno_audio_paths": ["missing.mp3"],
            }
        ],
        "negatives": [],
    }

    with pytest.raises(ValueError, match="audio file not found"):
        validate(raw, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Slow — end-to-end run against the live corpus. Opt-in via -m slow.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_run_eval_end_to_end_smoke(tmp_path):
    """If the corpus + a tiny synthetic golden set both exist, the eval runs to completion.

    This is a smoke test — verifies the orchestration doesn't blow up, not that
    the numbers are sensible. Skipped when the corpus is empty.
    """
    pytest.importorskip("librosa")
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    corpus_dir = repo_root / "quality-scorer" / "public" / "corpus"
    if not (corpus_dir / "embeddings.npy").exists():
        pytest.skip("corpus not populated; rebuild_corpus first")

    # The full run_eval main() requires a real golden set on disk — this
    # smoke test is intentionally a no-op skip-or-pass placeholder Codex can
    # implement against the local environment once the eval scripts work.
