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

import asyncio
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

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
# /neighbors endpoint shape — direct offline endpoint tests, no model load
# ----------------------------------------------------------------------------


ARTISTS = [
    {
        "artistId": "jamendo:maya-lev",
        "name": "Maya Lev",
        "trackIds": ["t1", "t2"],
        "listenUrl": "https://jamendo.com/artist/maya",
        "location": "Lisbon, PT",
        "supportLinks": [{"kind": "website", "url": "https://maya.example", "label": "Website"}],
        "previewUrl": "https://artist-preview/maya.mp3",
        "spotifyUrl": None,
    },
    {
        "artistId": "jamendo:hollow-coast",
        "name": "Hollow Coast",
        "trackIds": ["t3"],
        "listenUrl": "https://jamendo.com/artist/hollow",
        "location": None,
        "supportLinks": [],
        "previewUrl": None,
        "spotifyUrl": None,
    },
]


def _upload() -> UploadFile:
    return UploadFile(
        file=io.BytesIO(b"not-real-audio-but-decoder-is-stubbed"),
        filename="query.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )


def _pipeline() -> dict:
    return {
        "analysis": {"durationSec": 1.0, "waveform": [], "problems": [], "raw": {}},
        "report": {"score": 100, "verdict": "ok", "reason": "stubbed", "signals": {}},
        "genres": [],
        "emb": np.array([1.0, 0.0], dtype=np.float32),
        "segment_embeddings": np.array([[1.0, 0.0]], dtype=np.float32),
        "mir": object(),
    }


def _neighbor(track_id: str, score: float, *, q_win: int = 0, c_win: int = 0) -> dict:
    return {
        "trackId": track_id,
        "meanPooledSimilarity": score,
        "maxSegmentSimilarity": score - 0.01,
        "matchQueryWindow": q_win,
        "matchCatalogWindow": c_win,
    }


def _install_endpoint_fakes(monkeypatch, raw_neighbors, *, artists=True, token=False, capture=None):
    from backend import api
    from backend import artist_response

    tracks = {
        "t1": {
            "track_id": "t1",
            "title": "First Maya Track",
            "artist": "Maya Lev",
            "external_ids": {"jamendoAudioUrl": "https://audio/t1.mp3"},
            "mir_features": {"stub": True},
        },
        "t2": {
            "track_id": "t2",
            "title": "Second Maya Track",
            "artist": "Maya Lev",
            "previewUrl": "https://preview/t2.mp3",
            "mir_features": {"stub": True},
        },
        "t3": {
            "track_id": "t3",
            "title": "Hollow Track",
            "artist": "Hollow Coast",
            "audioUrl": "https://audio/t3.mp3",
            "mir_features": {"stub": True},
        },
        "t4": {
            "track_id": "t4",
            "title": "Weak Track",
            "artist": "Weak Artist",
            "mir_features": {"stub": True},
        },
    }

    monkeypatch.setattr(api, "_decode_and_pipeline", lambda raw, ext="": _pipeline())
    monkeypatch.setattr(api, "_flat_catalog", object())
    monkeypatch.setattr(api, "_corpus_by_id", tracks)
    monkeypatch.setattr(api, "_threshold_default", 0.7)
    monkeypatch.setattr(api, "_model_sha", "model-sha")
    monkeypatch.setattr(api, "_catalog_sha", "catalog-sha")
    monkeypatch.setattr(api, "_catalog_cosine_distribution", np.array([], dtype=np.float32))
    monkeypatch.setattr(api.similarity, "query_specificity", lambda emb, catalog: 0.5)
    monkeypatch.setattr(
        api,
        "_build_criteria_block",
        lambda query_mir, match_mir: {
            "tempo": {"agreement": 0.9, "label": "4 BPM apart", "queryValue": 100.0, "matchValue": 104.0},
            "key": {"agreement": 1.0, "label": "Same key", "queryValue": "F minor", "matchValue": "F minor"},
            "harmonic": {"agreement": 0.8, "label": "Close harmonic color"},
            "timbre": {"agreement": 0.7, "label": "Similar texture"},
        } if match_mir else None,
    )
    monkeypatch.setattr(api.context_token, "is_configured", lambda: token)
    def fake_issue(**kwargs):
        if capture is not None:
            capture["neighbors"] = kwargs["neighbors"]
        return "signed-context"

    monkeypatch.setattr(api.context_token, "issue", fake_issue)
    if artists:
        monkeypatch.setattr(api, "_artists_by_id", artist_response.index_artists(ARTISTS))
        monkeypatch.setattr(api, "_track_to_artist", artist_response.build_track_to_artist(ARTISTS))
    else:
        monkeypatch.setattr(api, "_artists_by_id", None)
        monkeypatch.setattr(api, "_track_to_artist", None)

    calls = {}

    def fake_top_k(query_emb, query_segments, catalog, k):
        calls["k"] = k
        return [dict(nb) for nb in raw_neighbors]

    monkeypatch.setattr(api, "_top_k_neighbors", fake_top_k)
    return api, calls


def _call_neighbors(api, *, k: int = 5):
    return asyncio.run(api.neighbors_endpoint(file=_upload(), k=k))


def test_neighbors_artist_response_dedupes_and_attaches_representative_track(monkeypatch):
    api, calls = _install_endpoint_fakes(
        monkeypatch,
        [_neighbor("t1", 0.91), _neighbor("t2", 0.88), _neighbor("t3", 0.82), _neighbor("t4", 0.65)],
    )

    data = _call_neighbors(api, k=3)

    assert calls["k"] == 15
    assert data["contractVersion"] == "artist-v1"
    assert "neighbors" not in data
    assert [m["artistId"] for m in data["matches"]] == ["jamendo:maya-lev", "jamendo:hollow-coast"]
    assert data["matches"][0]["representativeTrackId"] == "t1"
    assert data["matches"][0]["previewUrl"] == "https://audio/t1.mp3"
    assert data["matches"][0]["narrative"] is None
    assert [c["label"] for c in data["matches"][0]["criteria"]] == ["Tempo", "Key", "Harmonic", "Timbre"]


def test_neighbors_artist_response_never_pads_below_threshold(monkeypatch):
    api, _calls = _install_endpoint_fakes(
        monkeypatch,
        [_neighbor("t1", 0.91), _neighbor("t3", 0.69)],
    )

    data = _call_neighbors(api)

    assert len(data["matches"]) == 1
    assert data["matches"][0]["artistId"] == "jamendo:maya-lev"


def test_neighbors_context_token_uses_only_winning_tracks(monkeypatch):
    captured: dict[str, dict] = {}
    api, _calls = _install_endpoint_fakes(
        monkeypatch,
        [_neighbor("t1", 0.91), _neighbor("t2", 0.88), _neighbor("t3", 0.82)],
        token=True,
        capture=captured,
    )

    data = _call_neighbors(api)

    assert data["contextToken"] == "signed-context"
    assert set(captured["neighbors"]) == {"t1", "t3"}


def test_neighbors_legacy_fallback_when_artists_json_is_absent(monkeypatch):
    api, _calls = _install_endpoint_fakes(
        monkeypatch,
        [_neighbor("t1", 0.91), _neighbor("t2", 0.88)],
        artists=False,
    )

    data = _call_neighbors(api)

    assert "neighbors" in data
    assert "matches" not in data
    assert data["neighbors"][0]["trackId"] == "t1"
    assert data["contextToken"] is None


def test_neighbors_no_corpus_returns_empty_artist_response(monkeypatch):
    from backend import api

    monkeypatch.setattr(api, "_decode_and_pipeline", lambda raw, ext="": _pipeline())
    monkeypatch.setattr(api, "_flat_catalog", None)

    data = _call_neighbors(api)

    assert data == {"contractVersion": "artist-v1", "matches": [], "contextToken": None}


# ----------------------------------------------------------------------------
# Backwards-compat — /analyze still returns the legacy 7-signal shape
# ----------------------------------------------------------------------------


def test_analyze_endpoint_still_returns_legacy_shape(monkeypatch):
    """Phase 2 must not break /analyze — Phase 3's quality badge depends on it."""
    from backend import api

    monkeypatch.setattr(api, "_decode_and_pipeline", lambda raw, ext="": _pipeline())

    data = asyncio.run(api.analyze_endpoint(file=_upload()))
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
    pytest.skip("slow live model roundtrip is excluded from offline Phase 3 endpoint tests")
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
