"""Phase 3 core — turn ranked retrieval results into the artist-framed top-3.

Pure, side-effect-free mapping from (a) the ranked track matches the existing
similarity engine returns and (b) the enriched `artists.json` produced by Phase 2
(`scripts/phase2_catalog.serialize_artists`) into the frozen `ArtistMatch`
contract (`artist.py`).

The honest-threshold + no-padding + dedupe-by-artist rules (FR-9) live here so
they're unit-testable without booting the model or the endpoint. The `/neighbors`
endpoint (Phase 3 wiring) calls `build_artist_matches`; narrative + criteria are
attached by the endpoint later (both nullable in the contract).
"""
from __future__ import annotations

from typing import Iterable, Optional

from backend.artist import ArtistMatch, ArtistNeighborsResponse, SupportLink


def index_artists(artists: list[dict]) -> dict[str, dict]:
    """`artistId` -> artist record, from a loaded artists.json list."""
    return {a["artistId"]: a for a in artists if a.get("artistId")}


def build_track_to_artist(artists: list[dict]) -> dict[str, str]:
    """`trackId` -> `artistId`, so a matched track resolves to its artist."""
    idx: dict[str, str] = {}
    for a in artists:
        artist_id = a.get("artistId")
        if not artist_id:
            continue
        for tid in a.get("trackIds") or []:
            idx[tid] = artist_id
    return idx


def _artist_match(artist: dict, similarity: float) -> ArtistMatch:
    """Map one enriched artist record + the matched similarity to ArtistMatch.

    Optional fields (location, supportLinks, spotifyUrl, previewUrl) pass through
    only when present — `None`/`[]` otherwise, never a placeholder (FR-5).
    narrative + criteria are left empty here; the endpoint attaches them.
    """
    return ArtistMatch(
        artistId=artist["artistId"],
        name=artist.get("name") or "Unknown artist",
        similarity=float(similarity),
        previewUrl=artist.get("previewUrl"),
        listenUrl=artist.get("listenUrl") or "",
        location=artist.get("location"),
        supportLinks=[SupportLink(**link) for link in (artist.get("supportLinks") or [])],
        spotifyUrl=artist.get("spotifyUrl"),
        narrative=None,
        criteria=[],
    )


def build_artist_matches(
    ranked_tracks: Iterable[tuple[str, float]],
    artists_by_id: dict[str, dict],
    track_to_artist: dict[str, str],
    *,
    threshold: float,
    k: int = 3,
) -> list[ArtistMatch]:
    """The FR-9 core: top-k artists, deduped, above threshold, NEVER padded.

    ``ranked_tracks`` is (trackId, similarity) in DESCENDING similarity order
    (the similarity engine's output). We walk it once: keep the first (highest-
    similarity) track per artist, drop anything below ``threshold``, stop at k.
    Fewer than k honest matches is correct and expected — we never backfill with
    weak ones.
    """
    seen: set[str] = set()
    matches: list[ArtistMatch] = []
    for track_id, sim in ranked_tracks:
        if sim < threshold:
            continue  # honest threshold — below it never appears
        artist_id = track_to_artist.get(track_id)
        if not artist_id or artist_id in seen:
            continue  # one card per artist (dedupe), highest similarity wins
        artist = artists_by_id.get(artist_id)
        if not artist:
            continue
        seen.add(artist_id)
        matches.append(_artist_match(artist, sim))
        if len(matches) >= k:
            break
    return matches


def build_response(
    matches: list[ArtistMatch], context_token: Optional[str] = None
) -> ArtistNeighborsResponse:
    """Wrap matches in the frozen response envelope."""
    return ArtistNeighborsResponse(matches=matches, contextToken=context_token)
