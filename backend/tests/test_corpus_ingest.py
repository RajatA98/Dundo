"""Phase 1 — catalog ingest tests.

These tests codify PROJECT_PLAN Phase 1 acceptance criteria + the LOCKED
manifest schema. They are how Codex's implementation gets verified.

Run order of intent:
  - Fast unit tests run on every push (default `pytest`).
  - The slow CLAP-loading tests run only with `pytest -m slow` (mirror of the
    convention already in pyproject.toml).

Required files under `quality-scorer/public/corpus/` after a successful run
of `python -m backend.scripts.rebuild_corpus`:
  1. corpus.json
  2. embeddings.npy
  3. segment_embeddings.npz
  4. manifest.json
  5. examples.json

The corpus tests run against the REAL committed output files (they don't
re-run the ingest). The windowing tests do load CLAP and run a tiny audio
roundtrip — those are marked slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "quality-scorer" / "public" / "corpus"

REQUIRED_MANIFEST_FIELDS = {
    "model_id",
    "model_sha",
    "embedding_dim",
    "window_seconds",
    "query_max_seconds",
    "pooling",
    "threshold_default",
    "tier_counts",
    "generated_at",
    "sha256",
}

EXPECTED_EMBED_DIM = 512
EXPECTED_WINDOW_SECONDS = 10
EXPECTED_POOLING = "l2_normalized_mean"
EXPECTED_THRESHOLD_DEFAULT = 0.70
EXPECTED_QUERY_MAX_SECONDS = 90


def _corpus_present() -> bool:
    return all(
        (CORPUS_DIR / name).exists()
        for name in ("corpus.json", "embeddings.npy", "segment_embeddings.npz", "manifest.json", "examples.json")
    )


pytestmark_corpus = pytest.mark.skipif(
    not _corpus_present(),
    reason="corpus files not yet generated; run `python -m backend.scripts.rebuild_corpus`",
)


# ----------------------------------------------------------------------------
# Manifest schema (Codex review #4)
# ----------------------------------------------------------------------------


@pytestmark_corpus
def test_manifest_has_all_required_fields():
    """manifest.json must contain every field listed in PROJECT_PLAN Phase 1."""
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    assert not missing, f"manifest.json missing required fields: {missing}"


@pytestmark_corpus
def test_manifest_locked_constant_values():
    """Constants that must NOT drift between runs without an explicit decision."""
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    assert manifest["embedding_dim"] == EXPECTED_EMBED_DIM
    assert manifest["window_seconds"] == EXPECTED_WINDOW_SECONDS
    assert manifest["query_max_seconds"] == EXPECTED_QUERY_MAX_SECONDS
    assert manifest["pooling"] == EXPECTED_POOLING
    assert manifest["threshold_default"] == EXPECTED_THRESHOLD_DEFAULT


@pytestmark_corpus
def test_manifest_tier_counts_match_corpus():
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text())

    counted = {"tier1": 0, "tier2": 0}
    for row in corpus:
        counted[row["tier"]] = counted.get(row["tier"], 0) + 1

    assert manifest["tier_counts"] == counted


# ----------------------------------------------------------------------------
# embeddings.npy + segment_embeddings.npz (Codex review #1)
# ----------------------------------------------------------------------------


@pytestmark_corpus
def test_embeddings_shape_and_l2_normalized():
    """Mean-pooled embeddings must be (N, 512) float32 with rows L2-normalized."""
    arr = np.load(CORPUS_DIR / "embeddings.npy")
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text())

    assert arr.ndim == 2
    assert arr.shape == (len(corpus), EXPECTED_EMBED_DIM)
    assert arr.dtype == np.float32

    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "embeddings.npy rows must be L2-normalized"


@pytestmark_corpus
def test_segment_embeddings_keys_match_corpus_track_ids():
    """Every track in corpus.json must have a segment_embeddings entry, and vice versa."""
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text())
    seg = np.load(CORPUS_DIR / "segment_embeddings.npz")

    corpus_ids = {row["track_id"] for row in corpus}
    seg_keys = set(seg.files)

    assert corpus_ids == seg_keys, (
        f"track_id ↔ segment_embeddings mismatch. "
        f"In corpus but not segments: {corpus_ids - seg_keys}. "
        f"In segments but not corpus: {seg_keys - corpus_ids}."
    )


@pytestmark_corpus
def test_segment_embeddings_shape_and_l2_normalized():
    """Each segment_embeddings entry: (num_windows, 512), L2-normalized rows, num_windows ≥ 1."""
    seg = np.load(CORPUS_DIR / "segment_embeddings.npz")
    for track_id in seg.files:
        rows = seg[track_id]
        assert rows.ndim == 2, f"{track_id}: expected 2-D segment matrix, got {rows.shape}"
        assert rows.shape[1] == EXPECTED_EMBED_DIM, f"{track_id}: dim={rows.shape[1]}, expected {EXPECTED_EMBED_DIM}"
        assert rows.shape[0] >= 1, f"{track_id}: must have at least 1 segment"
        assert rows.dtype == np.float32, f"{track_id}: dtype={rows.dtype}, expected float32"
        norms = np.linalg.norm(rows, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"{track_id}: rows must be L2-normalized (got norms {norms})"


# ----------------------------------------------------------------------------
# corpus.json structure (Apple compliance + LOCKED_DECISIONS)
# ----------------------------------------------------------------------------


@pytestmark_corpus
def test_tier1_rows_have_attribution_and_track_view_url():
    """Apple Search API terms — every Tier-1 row must carry attribution + link-out."""
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text())
    tier1 = [r for r in corpus if r["tier"] == "tier1"]
    assert tier1, "expected at least one tier1 row in the corpus"

    for row in tier1:
        assert row.get("attribution_required") is True, f"{row['track_id']}: tier1 must set attribution_required=True"
        assert row.get("track_view_url"), f"{row['track_id']}: tier1 must include track_view_url"
        assert row["source"] == "itunes", f"{row['track_id']}: tier1.source must be 'itunes'"


@pytestmark_corpus
def test_tier2_rows_have_license_and_source_url():
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text())
    tier2 = [r for r in corpus if r["tier"] == "tier2"]
    if not tier2:
        pytest.skip("no tier2 tracks configured in catalog.yaml — skipping")

    for row in tier2:
        assert row.get("license_short"), f"{row['track_id']}: tier2 must include license_short"
        assert row.get("source_url"), f"{row['track_id']}: tier2 must include source_url"
        assert row["source"] in {"fma", "jamendo"}


@pytestmark_corpus
def test_no_audio_bytes_committed_to_corpus_dir():
    """Apple stream-not-cache + general no-bytes-shipped rule — no audio files in corpus dir."""
    audio_exts = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
    leaked = [p.name for p in CORPUS_DIR.iterdir() if p.suffix.lower() in audio_exts]
    assert not leaked, f"corpus directory must not contain audio bytes; found: {leaked}"


# ----------------------------------------------------------------------------
# examples.json (Phase 1 acceptance, both similarity metrics required)
# ----------------------------------------------------------------------------


@pytestmark_corpus
def test_examples_count_in_range():
    """Phase 1: 0–5 examples accepted (Suno audio lands in Phase 6). Phase 6: must be 3–5."""
    examples = json.loads((CORPUS_DIR / "examples.json").read_text())
    if not examples:
        pytest.skip(
            "no examples generated — example audio files arrive in Phase 6. "
            "build_examples() correctly skipped missing entries."
        )
    assert 3 <= len(examples) <= 5, f"expected 3–5 staged examples, got {len(examples)}"


@pytestmark_corpus
def test_examples_have_both_similarity_metrics():
    """Each example must carry both meanPooledSimilarity and maxSegmentSimilarity for its neighbors."""
    examples = json.loads((CORPUS_DIR / "examples.json").read_text())
    if not examples:
        pytest.skip("no examples generated yet — see Phase 6")
    for ex in examples:
        assert "neighbors" in ex and ex["neighbors"], f"{ex.get('id')}: missing neighbors"
        for nb in ex["neighbors"]:
            assert "meanPooledSimilarity" in nb, f"{ex['id']}: neighbor missing meanPooledSimilarity"
            assert "maxSegmentSimilarity" in nb, f"{ex['id']}: neighbor missing maxSegmentSimilarity"
        assert "verdictHeadline" in ex, f"{ex['id']}: missing verdictHeadline"


@pytestmark_corpus
def test_examples_completely_unique_copy_exact():
    """If any example is a Case-B (no match) example, the verdictHeadline string is locked."""
    examples = json.loads((CORPUS_DIR / "examples.json").read_text())
    if not examples:
        pytest.skip("no examples generated yet — see Phase 6")
    expected = "Completely unique — this track doesn't sound like anything in our reference catalog"
    case_b = [e for e in examples if "Completely unique" in e.get("verdictHeadline", "")]
    for ex in case_b:
        assert ex["verdictHeadline"] == expected, (
            f"{ex['id']}: empty-state copy must match LOCKED_DECISIONS exactly. "
            f"Got: {ex['verdictHeadline']!r}"
        )


# ----------------------------------------------------------------------------
# Windowed-encoding contract (also covers Phase 2 acceptance)
# Loads CLAP — marked slow.
# ----------------------------------------------------------------------------


@pytest.mark.slow
def test_windowed_encode_10s_matches_direct_encode():
    """10 s input → exactly 1 window; pooled output == direct encode within tolerance."""
    from backend import clap_engine, clap_windowed, config

    clap_engine.load()
    sr = config.CLAP_SR
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(10 * sr).astype(np.float32) * 0.1

    direct = clap_engine.encode_audio(wav, sr)
    pooled, segs = clap_windowed.encode_windowed(wav, sr)

    assert segs.shape == (1, config.CLAP_EMBED_DIM)
    assert np.allclose(pooled, direct, atol=1e-5), "10s pooled output must equal direct encode"


@pytest.mark.slow
def test_windowed_encode_30s_produces_three_l2_normalized_windows():
    """30 s input → exactly 3 windows; pooled output is L2-normalized mean."""
    from backend import clap_engine, clap_windowed, config

    clap_engine.load()
    sr = config.CLAP_SR
    rng = np.random.default_rng(1)
    wav = rng.standard_normal(30 * sr).astype(np.float32) * 0.1

    pooled, segs = clap_windowed.encode_windowed(wav, sr)

    assert segs.shape == (3, config.CLAP_EMBED_DIM)
    # rows L2-normalized
    assert np.allclose(np.linalg.norm(segs, axis=1), 1.0, atol=1e-4)
    # pooled is L2-normalized
    assert np.isclose(np.linalg.norm(pooled), 1.0, atol=1e-4)
    # pooled is the L2-normalized mean of segs (cosine ≈ 1 with the raw mean)
    raw_mean = segs.mean(axis=0)
    raw_mean /= np.linalg.norm(raw_mean) + 1e-12
    assert np.allclose(pooled, raw_mean, atol=1e-5)


@pytest.mark.slow
def test_windowed_encode_output_always_l2_normalized():
    """Any-length input: pooled output norm ≈ 1."""
    from backend import clap_engine, clap_windowed, config

    clap_engine.load()
    sr = config.CLAP_SR
    rng = np.random.default_rng(2)
    for seconds in (5, 10, 15, 30, 45, 60, 90):
        wav = rng.standard_normal(seconds * sr).astype(np.float32) * 0.1
        pooled, _ = clap_windowed.encode_windowed(wav, sr, max_seconds=config.CLAP_QUERY_MAX_SECONDS)
        assert np.isclose(np.linalg.norm(pooled), 1.0, atol=1e-4), f"{seconds}s pooled not L2-normalized"
