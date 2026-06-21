from __future__ import annotations

import json

from backend.artist import ArtistRecord
from backend.scripts.enrich_spike import (
    EnrichmentResult,
    FMAEnricher,
    JamendoEnricher,
    MusicBrainzEnricher,
    merge,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params or {}))
        if url.endswith("/artists/") and (params or {}).get("namesearch") == "Maya Lev":
            return _Response({"headers": {"status": "success"}, "results": [{"id": "42"}]})
        if url.endswith("/artists/"):
            return _Response(
                {
                    "headers": {"status": "success"},
                    "results": [
                        {
                            "id": "42",
                            "name": "Maya Lev",
                            "shareurl": "https://www.jamendo.com/artist/42/maya-lev",
                            "website": "https://mayalev.example",
                        }
                    ],
                }
            )
        if url.endswith("/artists/locations/"):
            # Real Jamendo shape: location is nested under results[0].locations[].
            return _Response(
                {
                    "headers": {"status": "success"},
                    "results": [{"id": "42", "locations": [{"city": "Lisbon", "country": "PT"}]}],
                }
            )
        raise AssertionError(f"unexpected call: {url} {params}")


def test_jamendo_enricher_resolves_artist_metadata_with_mocked_client(tmp_path):
    artist = ArtistRecord(
        artistId="jamendo:maya-lev",
        name="Maya Lev",
        source="jamendo",
        listenUrl="https://www.jamendo.com/track/1",
    )

    # Isolated cache dir so the test never reads/writes the real .cache.
    result = JamendoEnricher(client_id="test", client=_Client(), cache_dir=tmp_path).enrich(artist)

    assert result.sourceArtistId == "42"
    assert result.location == "Lisbon, PT"
    assert [(l.kind, l.url) for l in result.supportLinks] == [
        ("jamendo", "https://www.jamendo.com/artist/42/maya-lev"),
        ("website", "https://mayalev.example"),
    ]


def test_fma_enricher_reads_artist_metadata_csv(tmp_path):
    csv_path = tmp_path / "artists.csv"
    csv_path.write_text(
        "artist_name,artist_location,artist_website,artist_url,artist_donation_url\n"
        "Hollow Coast,\"Portland, United States\",https://hollow.example,"
        "https://freemusicarchive.org/music/Hollow_Coast,https://patreon.com/hollow\n"
    )

    artist = ArtistRecord(
        artistId="fma:hollow-coast",
        name="Hollow Coast",
        source="fma",
        listenUrl="https://freemusicarchive.org/music/Hollow_Coast",
    )

    result = FMAEnricher(fma_metadata_dir=tmp_path).enrich(artist)

    assert result.location == "Portland, United States"
    assert [(l.kind, l.url) for l in result.supportLinks] == [
        ("website", "https://hollow.example"),
        ("fma", "https://freemusicarchive.org/music/Hollow_Coast"),
        ("patreon", "https://patreon.com/hollow"),
    ]


def test_musicbrainz_enricher_requires_corroboration_for_spotify(tmp_path):
    dump = tmp_path / "musicbrainz_artists.json"
    dump.write_text(
        json.dumps(
            [
                {
                    "name": "Prata",
                    "mbid": "mbid-1",
                    "area": "Porto, Portugal",
                    "urls": [
                        {"type": "official homepage", "url": "https://prata.example"},
                        {"type": "bandcamp", "url": "https://prata.bandcamp.com"},
                        {"type": "spotify", "url": "https://open.spotify.com/artist/prata"},
                    ],
                },
                {
                    "name": "Name Only",
                    "mbid": "mbid-2",
                    "urls": [{"type": "spotify", "url": "https://open.spotify.com/artist/nameonly"}],
                },
            ]
        )
    )

    enricher = MusicBrainzEnricher(dump_path=dump)

    corroborated = enricher.enrich(ArtistRecord("jamendo:prata", "Prata", "jamendo"))
    assert corroborated.location == "Porto, Portugal"
    assert corroborated.spotifyUrl == "https://open.spotify.com/artist/prata"
    assert corroborated.spotifyConfidence == "mbid+external-link"
    assert [l.kind for l in corroborated.supportLinks] == ["website", "bandcamp"]

    name_only = enricher.enrich(ArtistRecord("jamendo:name-only", "Name Only", "jamendo"))
    assert name_only.spotifyUrl is None
    assert name_only.spotifyConfidence is None


def test_merge_keeps_spotify_confidence_gate():
    artist = ArtistRecord("jamendo:x", "X", "jamendo")
    merge(
        artist,
        [
            EnrichmentResult(spotifyUrl="https://open.spotify.com/artist/unverified"),
            EnrichmentResult(
                spotifyUrl="https://open.spotify.com/artist/verified",
                spotifyConfidence="mbid+external-link",
            ),
        ],
    )

    assert artist.spotifyUrl == "https://open.spotify.com/artist/verified"
