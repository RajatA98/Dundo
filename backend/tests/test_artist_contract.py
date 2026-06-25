"""Phase-1 contract tests — the artist entity, aggregation, and the guaranteed listen link.

These lock the *shape* Codex's enrichment and the Phase-3/4/5 work build against.
They do not test enrichment quality (that's measured by the coverage spike).
"""
from __future__ import annotations

from backend.artist import (
    CONTRACT_VERSION,
    ArtistMatch,
    ArtistNeighborsResponse,
    aggregate_tracks_by_artist,
    derive_host_listen_url,
)


def _track(tid, artist, source="jamendo", view_url=None, source_url=None):
    return {
        "track_id": tid,
        "artist": artist,
        "source": source,
        "track_view_url": view_url,
        "source_url": source_url,
    }


def test_contract_version_is_frozen_marker():
    # If this changes, the frontend (Phase 5) + narrative schema (Phase 4) must coordinate.
    assert CONTRACT_VERSION == "artist-v1"
    assert ArtistNeighborsResponse(matches=[]).contractVersion == "artist-v1"


def test_context_token_is_optional():
    # The existing signing path returns None when CONTEXT_TOKEN_HMAC_KEY is unset (dev mode).
    assert ArtistNeighborsResponse(matches=[]).contextToken is None
    assert ArtistNeighborsResponse(matches=[], contextToken="signed").contextToken == "signed"


def test_aggregation_groups_tracks_by_artist():
    tracks = [
        _track("jamendo:1", "Maya Lev", view_url="https://jamendo.com/track/1"),
        _track("jamendo:2", "Maya Lev", view_url="https://jamendo.com/track/2"),
        _track("jamendo:3", "Hollow Coast", view_url="https://jamendo.com/track/3"),
    ]
    records = aggregate_tracks_by_artist(tracks)
    assert set(records) == {"jamendo:maya-lev", "jamendo:hollow-coast"}
    assert records["jamendo:maya-lev"].trackIds == ["jamendo:1", "jamendo:2"]


def test_artist_id_is_source_scoped():
    # Same name on different sources must not collapse into one artist.
    records = aggregate_tracks_by_artist(
        [_track("j:1", "Echo", source="jamendo"), _track("f:1", "Echo", source="fma")]
    )
    assert set(records) == {"jamendo:echo", "fma:echo"}


def test_listen_url_is_guaranteed_when_source_url_present():
    # The "give them a listen" link must always resolve from the track's own fields.
    records = aggregate_tracks_by_artist(
        [_track("j:1", "Práta", view_url=None, source_url="https://jamendo.com/track/9")]
    )
    assert records["jamendo:práta"].listenUrl == "https://jamendo.com/track/9"


def test_derive_host_listen_url_prefers_view_url():
    assert derive_host_listen_url({"track_view_url": "A", "source_url": "B"}) == "A"
    assert derive_host_listen_url({"source_url": "B"}) == "B"
    assert derive_host_listen_url({}) == ""


def test_optional_fields_default_empty_not_placeholder():
    m = ArtistMatch(artistId="jamendo:x", name="X", similarity=0.8, listenUrl="u")
    assert m.location is None
    assert m.supportLinks == []
    assert m.spotifyUrl is None
    assert m.representativeTrackId is None
    # narrative + criteria are part of the contract but nullable/empty until populated.
    assert m.narrative is None
    assert m.criteria == []


def test_criteria_carry_data_not_presentation():
    # Contract holds the agreement value; the frontend derives bar width/color.
    m = ArtistMatch(
        artistId="jamendo:x", name="X", similarity=0.8, listenUrl="u",
        narrative="Both sit in a hushed F-minor.",
        criteria=[{"label": "Key", "detail": "Same key — F minor", "agreement": 1.0}],
    )
    assert m.narrative.startswith("Both")
    assert m.criteria[0].agreement == 1.0
    assert not hasattr(m.criteria[0], "fill")  # no presentation leak in the contract


def test_representative_track_id_serializes_without_version_bump():
    m = ArtistMatch(
        artistId="jamendo:x",
        name="X",
        similarity=0.8,
        listenUrl="u",
        representativeTrackId="jamendo:track:1",
    )
    data = ArtistNeighborsResponse(matches=[m]).model_dump()
    assert data["contractVersion"] == "artist-v1"
    assert data["matches"][0]["representativeTrackId"] == "jamendo:track:1"
