from __future__ import annotations

from backend.eval_metrics import cross_artist_tag_overlap


def _catalog_tags():
    return {
        "q": {"coarseGenre": ["rock"], "mood": ["energetic"]},
        "n1": {"coarseGenre": ["rock"], "mood": []},
        "n2": {"coarseGenre": ["jazz"], "mood": []},
        "n3": {"coarseGenre": ["classical"], "mood": []},
        "r1": {"coarseGenre": ["jazz"], "mood": []},
        "r2": {"coarseGenre": ["classical"], "mood": []},
    }


def test_detects_shared_genre_with_genuinely_different_artist_neighbor():
    loo_results = [
        {
            "query_track_id": "q",
            "target_track_ids": [],
            "top_neighbors": [
                {"trackId": "n1", "meanPooledSimilarity": 0.9},
                {"trackId": "n2", "meanPooledSimilarity": 0.8},
                {"trackId": "n3", "meanPooledSimilarity": 0.7},
            ],
        }
    ]
    result = cross_artist_tag_overlap(loo_results, _catalog_tags(), top_k=3, random_seed=1)
    assert result["n"] == 1
    assert result["skippedNoTags"] == 0
    assert result["crossArtistTagOverlapAt3"] == 1.0
    assert 0.0 <= result["randomBaselineTagOverlapAt3"] <= 1.0


def test_same_artist_neighbors_are_excluded_not_counted_as_a_match():
    loo_results = [
        {
            "query_track_id": "q",
            # n1 shares the query's artist (it's in target_track_ids) — must
            # be excluded from the cross-artist check even though it shares
            # a tag with the query.
            "target_track_ids": ["n1"],
            "top_neighbors": [
                {"trackId": "n1", "meanPooledSimilarity": 0.95},
                {"trackId": "n2", "meanPooledSimilarity": 0.8},
                {"trackId": "n3", "meanPooledSimilarity": 0.7},
            ],
        }
    ]
    result = cross_artist_tag_overlap(loo_results, _catalog_tags(), top_k=3, random_seed=1)
    assert result["n"] == 1
    assert result["crossArtistTagOverlapAt3"] == 0.0


def test_query_with_no_real_tags_is_skipped_not_counted_as_a_miss():
    loo_results = [
        {
            "query_track_id": "untagged",
            "target_track_ids": [],
            "top_neighbors": [{"trackId": "n1", "meanPooledSimilarity": 0.9}],
        }
    ]
    result = cross_artist_tag_overlap(loo_results, _catalog_tags(), top_k=3, random_seed=1)
    assert result["n"] == 0
    assert result["skippedNoTags"] == 1
    assert result["crossArtistTagOverlapAt3"] is None


def test_no_loo_results_returns_none_metrics_not_a_crash():
    result = cross_artist_tag_overlap([], _catalog_tags(), top_k=3, random_seed=1)
    assert result == {
        "n": 0,
        "skippedNoTags": 0,
        "crossArtistTagOverlapAt3": None,
        "randomBaselineTagOverlapAt3": None,
    }
