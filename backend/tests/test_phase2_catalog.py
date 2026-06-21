from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from backend.artist import ArtistRecord, SupportLink
from backend.scripts.phase2_catalog import (
    STATUS_PENDING,
    STATUS_SKIPPED_DUPLICATE,
    STATUS_SKIPPED_LICENSE,
    STATUS_SKIPPED_PER_ARTIST_CAP,
    build_checkpoint_manifest,
    serialize_artists,
    validate_manifest_file_hashes,
    validate_phase2_catalog_config,
)
from backend.scripts.verify_matching import _select_targets


@dataclass
class Candidate:
    jamendo_track_id: str
    title: str
    artist: str
    license_short: str = "MTG-Jamendo (Creative Commons)"
    source_url: str = ""
    audio_path_or_url: str = ""


def test_phase2_config_rejects_retired_sources():
    with pytest.raises(ValueError, match="tier1"):
        validate_phase2_catalog_config({"tier1": [{"title": "old"}], "tier2": {"jamendo": {"count": 1}}})

    with pytest.raises(ValueError, match="Jamendo-only"):
        validate_phase2_catalog_config({"tier2": {"jamendo": {"count": 1}, "fma": {"count": 1}}})

    assert validate_phase2_catalog_config({"tier2": {"jamendo": {"count": 55000}}})["count"] == 55000


def test_checkpoint_manifest_records_quality_gate_statuses():
    tracks = [
        Candidate("1", "A", "Artist", source_url="https://jamendo.com/track/1"),
        Candidate("1", "A duplicate", "Artist", source_url="https://jamendo.com/track/1b"),
        Candidate("2", "Bad license", "Other", license_short="All rights reserved", source_url="https://jamendo.com/track/2"),
        Candidate("3", "Cap", "Artist", source_url="https://jamendo.com/track/3"),
    ]

    manifest = build_checkpoint_manifest(tracks, per_artist_cap=1, generated_at="2026-06-21T00:00:00+00:00")

    entries = manifest["entries"]
    assert entries["jamendo:1"]["status"] == STATUS_SKIPPED_DUPLICATE
    assert entries["jamendo:2"]["status"] == STATUS_SKIPPED_LICENSE
    assert entries["jamendo:3"]["status"] == STATUS_SKIPPED_PER_ARTIST_CAP
    assert manifest["summary"]["byStatus"] == {
        STATUS_SKIPPED_DUPLICATE: 1,
        STATUS_SKIPPED_LICENSE: 1,
        STATUS_SKIPPED_PER_ARTIST_CAP: 1,
    }


def test_checkpoint_manifest_accepts_pending_unique_cc_tracks():
    manifest = build_checkpoint_manifest(
        [
            Candidate("1", "A", "One", source_url="https://jamendo.com/track/1"),
            Candidate("2", "B", "One", source_url="https://jamendo.com/track/2"),
        ],
        per_artist_cap=2,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert [entry["status"] for entry in manifest["entries"].values()] == [STATUS_PENDING, STATUS_PENDING]
    assert manifest["summary"]["byStatus"] == {STATUS_PENDING: 2}


def test_serialize_artists_writes_artist_entity_shape():
    tracks = [
        {
            "track_id": "tier2:jamendo:1",
            "artist": "Maya Lev",
            "source": "jamendo",
            "source_url": "https://www.jamendo.com/track/1",
            "external_ids": {"jamendoAudioUrl": "https://audio.example/1.mp3"},
        }
    ]
    enriched = ArtistRecord(
        artistId="jamendo:maya-lev",
        name="Maya Lev",
        source="jamendo",
        trackIds=["tier2:jamendo:1"],
        representativeTrackId="tier2:jamendo:1",
        sourceArtistId="42",
        listenUrl="https://www.jamendo.com/artist/42/maya-lev",
        location="Lisbon, PT",
        supportLinks=[SupportLink(kind="jamendo", url="https://www.jamendo.com/artist/42/maya-lev", label="Jamendo")],
    )

    artists = serialize_artists(tracks, {"jamendo:maya-lev": enriched})

    assert artists == [
        {
            "artistId": "jamendo:maya-lev",
            "name": "Maya Lev",
            "source": "jamendo",
            "trackIds": ["tier2:jamendo:1"],
            "representativeTrackId": "tier2:jamendo:1",
            "sourceArtistId": "42",
            "listenUrl": "https://www.jamendo.com/artist/42/maya-lev",
            "location": "Lisbon, PT",
            "supportLinks": [
                {"kind": "jamendo", "url": "https://www.jamendo.com/artist/42/maya-lev", "label": "Jamendo"}
            ],
            "spotifyUrl": None,
            "previewUrl": "https://audio.example/1.mp3",
        }
    ]


def test_validate_manifest_file_hashes_detects_mismatch(tmp_path):
    artifact = tmp_path / "corpus.json"
    artifact.write_text("old")
    manifest = {
        "sha256": {
            "files": {
                "corpus.json": "0000",
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checksum"):
        validate_manifest_file_hashes(tmp_path)


def test_verify_matching_selects_jamendo_audio_targets():
    catalog = [
        {"track_id": "tier1:itunes:1", "source": "itunes", "external_ids": {"previewUrl": "https://apple.example/a.m4a"}},
        {"track_id": "tier2:jamendo:1", "source": "jamendo", "external_ids": {}},
        {
            "track_id": "tier2:jamendo:2",
            "source": "jamendo",
            "external_ids": {"jamendoAudioUrl": "https://audio.example/2.mp3"},
        },
    ]

    assert [t["track_id"] for t in _select_targets(catalog, None)] == ["tier2:jamendo:2"]
    assert [t["track_id"] for t in _select_targets(catalog, "tier2:jamendo:2")] == ["tier2:jamendo:2"]
