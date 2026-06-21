"""Phase 3 contract tests — the artist-match builder (FR-9 honest threshold / no padding)."""
from __future__ import annotations

from backend.artist_response import (
    build_artist_matches,
    build_track_to_artist,
    index_artists,
)


ARTISTS = [
    {
        "artistId": "jamendo:maya-lev",
        "name": "Maya Lev",
        "trackIds": ["t1", "t2"],
        "listenUrl": "https://jamendo.com/artist/maya",
        "location": "Lisbon, PT",
        "supportLinks": [{"kind": "website", "url": "https://maya.example", "label": "Website"}],
        "previewUrl": "https://stream/maya",
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


def _indexes():
    return index_artists(ARTISTS), build_track_to_artist(ARTISTS)


def test_dedupes_to_one_card_per_artist_highest_similarity_wins():
    by_id, t2a = _indexes()
    # t1 and t2 are both Maya, t1 ranked higher → one Maya card at 0.9
    ranked = [("t1", 0.9), ("t2", 0.85), ("t3", 0.8)]
    matches = build_artist_matches(ranked, by_id, t2a, threshold=0.7, k=3)
    assert [m.artistId for m in matches] == ["jamendo:maya-lev", "jamendo:hollow-coast"]
    assert matches[0].similarity == 0.9  # highest Maya track, not 0.85


def test_threshold_excludes_weak_and_never_pads():
    by_id, t2a = _indexes()
    # only t1 (Maya) crosses 0.7; t3 is below → exactly ONE honest match, not padded to 3
    ranked = [("t1", 0.88), ("t3", 0.42)]
    matches = build_artist_matches(ranked, by_id, t2a, threshold=0.7, k=3)
    assert len(matches) == 1
    assert matches[0].artistId == "jamendo:maya-lev"


def test_caps_at_k():
    by_id, t2a = _indexes()
    ranked = [("t1", 0.9), ("t3", 0.8)]
    matches = build_artist_matches(ranked, by_id, t2a, threshold=0.7, k=1)
    assert len(matches) == 1


def test_optional_fields_passthrough_only_when_present():
    by_id, t2a = _indexes()
    matches = build_artist_matches([("t1", 0.9), ("t3", 0.8)], by_id, t2a, threshold=0.7)
    maya, hollow = matches
    # Maya has location + a support link; Hollow has neither → empty/None, no placeholder
    assert maya.location == "Lisbon, PT"
    assert [l.kind for l in maya.supportLinks] == ["website"]
    assert maya.previewUrl == "https://stream/maya"
    assert hollow.location is None
    assert hollow.supportLinks == []
    assert hollow.previewUrl is None


def test_narrative_and_criteria_empty_until_endpoint_attaches():
    by_id, t2a = _indexes()
    m = build_artist_matches([("t1", 0.9)], by_id, t2a, threshold=0.7)[0]
    assert m.narrative is None
    assert m.criteria == []


def test_listen_url_always_present():
    by_id, t2a = _indexes()
    for m in build_artist_matches([("t1", 0.9), ("t3", 0.8)], by_id, t2a, threshold=0.7):
        assert m.listenUrl  # guaranteed — the "give them a listen" action
