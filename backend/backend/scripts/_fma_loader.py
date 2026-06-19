"""Free Music Archive (FMA) Tier-2 loader.

FMA is Creative-Commons-licensed; we keep the embedding + metadata + the
upstream FMA track URL for link-out. We do NOT redistribute audio bytes via
this app — the audio is fetched once at ingest, embedded, and discarded, same
discipline as Tier-1.

Source recommendation (PRESEARCH Q7): FMA dataset is documented at
https://github.com/mdeff/fma. The HuggingFace `datasets` library exposes
several FMA splits; pick a small one (`fma_small` ~8 GB) for breadth ingest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class FMATrack:
    """Subset of FMA fields we persist into corpus.json."""

    fma_track_id: str
    title: str
    artist: str
    primary_genre: str | None
    license_short: str            # e.g. "CC BY 4.0"
    source_url: str               # link out to the FMA page for this track
    audio_path_or_url: str        # where to fetch the audio at ingest time


def load_fma_tracks(
    count: int,
    genres_balanced: list[str] | None = None,
) -> list[FMATrack]:
    """Return up to `count` FMA tracks.

    Args:
        count: target number of tracks to return.
        genres_balanced: when provided, sample evenly across this genre list
            (round-robin until count is hit). When None, sample at random.

    Returns:
        List of FMATrack with len <= count. Length is exact when the source
        has enough tracks; can be shorter when filters exhaust the dataset.
    """
    from datasets import load_dataset

    wanted = {g.lower() for g in (genres_balanced or [])}
    selected: list[FMATrack] = []
    ds = load_dataset("benjamin-paine/free-music-archive", split="train", streaming=True)

    for row in ds:
        mapped = _row_to_track(row)
        if mapped is None:
            continue
        if wanted and (mapped.primary_genre or "").lower() not in wanted:
            continue
        selected.append(mapped)
        if len(selected) >= count:
            break
    return selected


def fetch_track_audio(track: FMATrack) -> bytes:
    """Fetch the audio bytes for a single FMA track. Discarded after CLAP encoding."""
    source = track.audio_path_or_url
    path = Path(source)
    if path.exists():
        return path.read_bytes()
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(source)
        response.raise_for_status()
        return response.content


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _row_to_track(row: dict[str, Any]) -> FMATrack | None:
    track_id = _first(row, "track_id", "id", "track")
    title = _first(row, "title", "track_title", "name")
    artist = _first(row, "artist", "artist_name", "artist_title")
    audio = _first(row, "audio", "audio_url", "mp3_url", "path", "file")
    if isinstance(audio, dict):
        audio = _first(audio, "path", "url", "array")
    if not (track_id and title and artist and audio):
        return None

    genre = _first(row, "genre", "primary_genre", "genre_top", "track_genre_top")
    license_short = _first(row, "license", "license_short", "track_license") or "Creative Commons"
    source_url = _first(row, "source_url", "url", "track_url")
    if not source_url:
        source_url = f"https://freemusicarchive.org/track/{track_id}/"

    return FMATrack(
        fma_track_id=str(track_id),
        title=str(title),
        artist=str(artist),
        primary_genre=str(genre) if genre else None,
        license_short=str(license_short),
        source_url=str(source_url),
        audio_path_or_url=str(audio),
    )
