"""MTG-Jamendo Tier-2 loader (alternative breadth source to FMA).

Same Creative-Commons discipline as FMA loader. Use one or the other (or both)
based on what `catalog.yaml` says. Per PRESEARCH Q7, MTG-Jamendo is
55k+ CC-licensed full tracks with richer genre tagging than FMA, but
recognizability is also near zero — same trade-off.

Source: https://mtg.github.io/mtg-jamendo-dataset/
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import math
from urllib.parse import urljoin

import httpx


@dataclass
class JamendoTrack:
    """Subset of Jamendo fields we persist into corpus.json."""

    jamendo_track_id: str
    title: str
    artist: str
    primary_genre: str | None
    license_short: str
    source_url: str
    audio_path_or_url: str


def load_jamendo_tracks(
    count: int,
    genres_balanced: list[str] | None = None,
) -> list[JamendoTrack]:
    """Return up to `count` MTG-Jamendo tracks.

    Same signature/semantics as `_fma_loader.load_fma_tracks`.
    """
    base_url = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/"
    metadata_url = urljoin(base_url, "autotagging.tsv")
    wanted = {_normalize_genre(g) for g in (genres_balanced or [])}
    target_candidates = max(count, math.ceil(count * 1.5))

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(metadata_url)
        response.raise_for_status()
        metadata_text = response.text
        resolved = _resolve_metadata_pointer(metadata_text)
        if resolved:
            response = client.get(urljoin(base_url, resolved))
            response.raise_for_status()
            metadata_text = response.text

    rows = csv.DictReader(io.StringIO(metadata_text.lstrip("\ufeff")), delimiter="\t")
    selected: list[JamendoTrack] = []
    by_genre: dict[str, list[JamendoTrack]] = {g: [] for g in wanted}
    fallback: list[JamendoTrack] = []
    seen: set[str] = set()
    for row in rows:
        normalized_row = {_clean_key(k): v for k, v in row.items() if k is not None}
        track_id = normalized_row.get("track_id") or normalized_row.get("track")
        if not track_id:
            continue
        track_key = str(track_id)
        if track_key in seen:
            continue

        tags = _row_tags(normalized_row)
        genres = _genres_from_tags(tags)
        primary_genre = _display_genre(genres[0]) if genres else None
        if wanted and not (set(genres) & wanted):
            continue

        seen.add(track_key)
        numeric_id = _numeric_track_id(track_key)
        track = JamendoTrack(
            jamendo_track_id=numeric_id,
            title=normalized_row.get("title") or f"Jamendo {numeric_id}",
            artist=normalized_row.get("artist") or normalized_row.get("artist_id") or "Unknown artist",
            primary_genre=primary_genre,
            license_short=normalized_row.get("license") or "MTG-Jamendo (Creative Commons)",
            source_url=f"https://www.jamendo.com/track/{numeric_id}",
            audio_path_or_url=f"https://mp3l.jamendo.com/?trackid={numeric_id}&format=mp32",
        )

        matched = set(genres) & wanted
        if matched:
            by_genre[sorted(matched)[0]].append(track)
        else:
            fallback.append(track)

    if wanted:
        selected = _round_robin(by_genre, target_candidates)
        if len(selected) < target_candidates:
            selected.extend(t for t in fallback if t.jamendo_track_id not in {s.jamendo_track_id for s in selected})
            selected = selected[:target_candidates]
    else:
        selected = fallback[:target_candidates]

    return selected


def _round_robin(groups: dict[str, list[JamendoTrack]], limit: int) -> list[JamendoTrack]:
    selected: list[JamendoTrack] = []
    seen: set[str] = set()
    keys = sorted(groups)
    index = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            bucket = groups[key]
            if index >= len(bucket):
                continue
            track = bucket[index]
            if track.jamendo_track_id not in seen:
                selected.append(track)
                seen.add(track.jamendo_track_id)
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


def fetch_track_audio(track: JamendoTrack) -> bytes:
    """Fetch the audio bytes for a single Jamendo track. Discarded after CLAP encoding."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(track.audio_path_or_url)
        if response.status_code == 429 and "format=mp32" in track.audio_path_or_url:
            fallback_url = track.audio_path_or_url.replace("format=mp32", "format=mp31")
            response = client.get(fallback_url)
        response.raise_for_status()
        return response.content


def _clean_key(key: str) -> str:
    return key.strip().lower().lstrip("\ufeff")


def _resolve_metadata_pointer(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    if "\n" not in stripped and stripped.endswith(".tsv"):
        return stripped
    return None


def _row_tags(row: dict[str, str]) -> list[str]:
    tags = row.get("tags") or ""
    extras = row.get("") or row.get(None) or []
    if isinstance(extras, str):
        extras = [extras]
    return [tag for tag in [tags, *extras] if tag]


def _genres_from_tags(tags: list[str]) -> list[str]:
    genres: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        for tag in raw.replace(",", "\t").split("\t"):
            tag = tag.strip()
            if not tag.startswith("genre---"):
                continue
            genre = _normalize_genre(tag.removeprefix("genre---"))
            if genre not in seen:
                genres.append(genre)
                seen.add(genre)
    return genres


def _normalize_genre(genre: str) -> str:
    return genre.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _display_genre(genre: str) -> str:
    return genre.replace("hiphop", "hip-hop").title()


def _numeric_track_id(track_id: str) -> str:
    if track_id.startswith("track_"):
        return str(int(track_id.removeprefix("track_")))
    return track_id.lstrip("0") or "0"
