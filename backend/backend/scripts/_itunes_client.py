"""iTunes Search API client for Tier-1 catalog ingest.

Per LOCKED_DECISIONS:
  - No auth required (we use the public Search endpoint).
  - Soft rate limit ~20 req/min. Batch with sleeps.
  - Apple stream-not-cache rule: fetch the preview ONLY long enough to compute
    the embedding, then discard. Never re-host the bytes.
  - UI must show iTunes attribution + link out to `trackViewUrl`; this client
    surfaces both fields so the corpus_writer can stamp `attributionRequired`.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_RATE_LIMIT_PER_MINUTE = 18  # under the ~20/min soft cap with headroom


@dataclass
class ITunesTrack:
    """Subset of iTunes Search API fields we persist into corpus.json."""

    track_id: int
    artist_id: int
    collection_id: int | None
    track_name: str
    artist_name: str
    collection_name: str | None
    preview_url: str           # 30 s AAC; fetched once at ingest, never stored
    track_view_url: str        # link-out required for Apple compliance
    primary_genre_name: str | None
    release_date: str | None   # ISO 8601 string
    track_time_millis: int | None
    artwork_url_100: str | None


def search_track(title: str, artist: str, country: str = "US") -> ITunesTrack | None:
    """Look up (title, artist) on the iTunes Search API and return the top hit.

    Sends a GET to `ITUNES_SEARCH_URL` with the documented params:
      - term=<title> <artist>
      - media=music
      - entity=song
      - limit=1
      - country=<country>

    Returns the top result parsed into ITunesTrack, or None when iTunes returns
    `resultCount: 0`. Raises `httpx.HTTPStatusError` on any non-2xx.

    Args:
        title: track title; URL-encoding handled by httpx.
        artist: artist name; URL-encoding handled by httpx.
        country: 2-letter iTunes storefront. Default US.
    """
    params = {
        "term": f"{title} {artist}",
        "media": "music",
        "entity": "song",
        "limit": 1,
        "country": country,
    }
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        response = client.get(ITUNES_SEARCH_URL, params=params)
        response.raise_for_status()
    body = response.json()
    if int(body.get("resultCount", 0)) == 0 or not body.get("results"):
        return None

    row = body["results"][0]
    preview_url = row.get("previewUrl")
    track_view_url = row.get("trackViewUrl")
    if not preview_url or not track_view_url:
        return None

    return ITunesTrack(
        track_id=int(row["trackId"]),
        artist_id=int(row["artistId"]),
        collection_id=row.get("collectionId"),
        track_name=row.get("trackName") or title,
        artist_name=row.get("artistName") or artist,
        collection_name=row.get("collectionName"),
        preview_url=preview_url,
        track_view_url=track_view_url,
        primary_genre_name=row.get("primaryGenreName"),
        release_date=row.get("releaseDate"),
        track_time_millis=row.get("trackTimeMillis"),
        artwork_url_100=row.get("artworkUrl100"),
    )


def fetch_preview(preview_url: str) -> bytes:
    """Fetch the iTunes 30 s preview AAC bytes for embedding.

    Streamed and discarded by the caller after CLAP encoding (Apple terms —
    NEVER persist these bytes anywhere outside transient memory).

    Args:
        preview_url: the `previewUrl` from `ITunesTrack`.

    Returns:
        Raw AAC bytes. Typical size ~600 KB.

    Raises:
        httpx.HTTPStatusError on non-2xx.
    """
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        response = client.get(preview_url)
        response.raise_for_status()
        return response.content


def rate_limited_iterator(items: list[Any], per_minute: int = ITUNES_RATE_LIMIT_PER_MINUTE):
    """Yield items at most `per_minute` per minute, sleeping when needed.

    Used by `rebuild_corpus.py` to throttle the Tier-1 search loop under Apple's
    ~20/min soft cap.
    """
    delay = 60.0 / max(1, per_minute)
    last_yield: float | None = None
    for item in items:
        if last_yield is not None:
            elapsed = time.monotonic() - last_yield
            if elapsed < delay:
                time.sleep(delay - elapsed)
        yield item
        last_yield = time.monotonic()
