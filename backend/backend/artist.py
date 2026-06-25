"""Artist entity + the frozen, artist-framed `/neighbors` response contract (Phase 1).

This is the LOAD-BEARING contract for Dundo v1. The frontend (Phase 5) and the
narrative schema (Phase 4) both build against `ArtistNeighborsResponse`, so treat
`CONTRACT_VERSION` as frozen: bump the version (and coordinate frontend + narrative)
rather than silently renaming fields or changing shapes.

Scaffolded by Claude: the Pydantic models, the deterministic track->artist
aggregation, and the guaranteed host-platform "listen" link. Enrichment of
`location` / `supportLinks` / `spotifyUrl` is implemented by Codex via the
providers in ``backend/scripts/enrich_spike.py`` — see
``factory/artifacts/CODEX_PHASE_1_IMPLEMENTATION.md``.

Design rules baked into the contract (from PRD.md + PRESEARCH.md):
  * The artist is the hero; ``similarity`` is a quiet rank signal, never the headline.
  * ``listenUrl`` is GUARANTEED (every match has a working "give them a listen" link).
  * ``location`` / ``supportLinks`` / ``spotifyUrl`` are OPTIONAL — render only when
    present, never an empty placeholder (FR-5). Enrichment coverage is partial:
    listen link ~90%+, location ~40-55% (PRESEARCH.md).
  * ``spotifyUrl`` is only ever set on a confident identity match (PRESEARCH/Codex #5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

# Bump this (and coordinate frontend + narrative schema) on any breaking shape change.
CONTRACT_VERSION = "artist-v1"

SupportKind = Literal[
    "jamendo", "fma", "bandcamp", "patreon", "website", "spotify", "other"
]


class SupportLink(BaseModel):
    """An outbound link a user can act on to support the artist."""

    kind: SupportKind
    url: str
    label: str  # display label, e.g. "Bandcamp", "Support on Patreon"


class Criterion(BaseModel):
    """One of the four MIR agreement criteria (ADR-0004), as data — the frontend
    derives the bar width/color from ``agreement`` (no presentation in the contract)."""

    label: str  # "Tempo" | "Key" | "Harmonic" | "Timbre"
    detail: str  # human label, e.g. "4 BPM apart", "Same key — F minor"
    agreement: float = Field(..., ge=0.0, le=1.0, description="0–1 closeness on this criterion")


class ArtistMatch(BaseModel):
    """One of the top-3 artist matches, framed around the human artist."""

    artistId: str = Field(..., description="Stable, source-scoped id, e.g. 'jamendo:maya-lev'")
    name: str
    similarity: float = Field(
        ..., description="Cosine rank signal. Rendered quiet/secondary, never the headline %."
    )
    previewUrl: Optional[str] = Field(None, description="Artist track preview audio URL")
    listenUrl: str = Field(
        ..., description="GUARANTEED 'give them a listen' link — host page (Jamendo/FMA/Bandcamp)"
    )
    location: Optional[str] = Field(None, description="'City, Country' when known (~half coverage)")
    supportLinks: list[SupportLink] = Field(
        default_factory=list, description="Optional extra support links (Bandcamp/Patreon/site)"
    )
    spotifyUrl: Optional[str] = Field(
        None, description="Optional secondary 'Listen on Spotify' — only on a confident match"
    )
    # --- presentation fields the card renders inline (populated at /neighbors time) ---
    representativeTrackId: Optional[str] = Field(
        None,
        description="Winning catalog track id for this artist match; frontend passes it to /narrative.",
    )
    narrative: Optional[str] = Field(
        None,
        description="Grounded 'why this resonates' paragraph (ADR-0005). NULLABLE — null when "
        "/narrative is disabled (no OPENAI_API_KEY) or not yet hydrated; card degrades gracefully.",
    )
    criteria: list[Criterion] = Field(
        default_factory=list,
        description="4-criterion MIR evidence (ADR-0004), computed at /neighbors time. Empty when uncomputed.",
    )
    # NOTE: these are NOT produced by the enrichment providers (enrich_spike.py) — they come
    # from the existing mir_features / rag_narrative machinery when the endpoint is wired (Phase 3).
    # Enrichment only fills listenUrl / location / supportLinks / spotifyUrl.


class ArtistNeighborsResponse(BaseModel):
    """The artist-framed top-3 response. Replaces the legacy track-shaped payload."""

    contractVersion: str = Field(default=CONTRACT_VERSION)
    matches: list[ArtistMatch] = Field(
        ..., description="Top 3 — or fewer honest matches (1-2). NEVER padded with weak matches (FR-9)."
    )
    # Optional: the existing context-signing path returns None when CONTEXT_TOKEN_HMAC_KEY is unset
    # (dev / no-narrative mode). Keep nullable rather than forcing signing to always be configured.
    contextToken: Optional[str] = Field(
        None, description="HMAC-signed token carrying per-match narrative context; null when signing is unconfigured"
    )


# --------------------------------------------------------------------------------
# Catalog-side aggregation (deterministic; no external calls). Enrichment fields
# below are populated by Codex's providers in enrich_spike.py.
# --------------------------------------------------------------------------------


@dataclass
class ArtistRecord:
    """All catalog tracks by one artist, aggregated. The unit enrichment operates on."""

    artistId: str
    name: str
    source: str  # "jamendo" | "fma" | ...
    trackIds: list[str] = field(default_factory=list)
    representativeTrackId: Optional[str] = None
    # Real source artist id (e.g. Jamendo artist id) when resolvable — hardens against
    # name collisions. CODEX: populate via the Jamendo API where available.
    sourceArtistId: Optional[str] = None

    # --- enrichment outputs (CODEX fills these via enrich_spike providers) ---
    listenUrl: Optional[str] = None  # scaffold sets a guaranteed baseline; CODEX may upgrade to artist page
    location: Optional[str] = None
    supportLinks: list[SupportLink] = field(default_factory=list)
    spotifyUrl: Optional[str] = None


def _slug(name: str) -> str:
    # Unicode-aware: preserve accented/non-Latin letters (common in CC indie catalogs)
    # rather than stripping them. ``\W`` is unicode-aware for str in Python 3.
    s = re.sub(r"[\W_]+", "-", (name or "").strip().lower()).strip("-")
    return s or "unknown"


def aggregate_tracks_by_artist(tracks: list[dict]) -> dict[str, ArtistRecord]:
    """Group corpus tracks into ``ArtistRecord``s keyed by ``source:slug(name)``.

    Deterministic and side-effect free. v1 ``artistId`` is ``f"{source}:{slug(name)}"``.

    CODEX: where a real source artist id exists (Jamendo artist id via the API),
    set ``sourceArtistId`` and prefer it for identity, to harden against the
    name-collision risk PRESEARCH flagged.
    """
    records: dict[str, ArtistRecord] = {}
    for t in tracks:
        name = (t.get("artist") or "Unknown artist").strip()
        source = t.get("source") or "unknown"
        artist_id = f"{source}:{_slug(name)}"
        rec = records.get(artist_id)
        if rec is None:
            rec = ArtistRecord(
                artistId=artist_id,
                name=name,
                source=source,
                representativeTrackId=t.get("track_id"),
                listenUrl=derive_host_listen_url(t),  # guaranteed baseline
            )
            records[artist_id] = rec
        if t.get("track_id"):
            rec.trackIds.append(t["track_id"])
    return records


def derive_host_listen_url(track: dict) -> str:
    """Guaranteed baseline 'give them a listen' link, from the track's own source fields.

    No network call — every Jamendo/FMA track carries its own page URL, so this
    guarantees a working listen link even before enrichment. CODEX may upgrade this
    to the artist's *page* (vs. a track page) and add richer support links.
    """
    return track.get("track_view_url") or track.get("source_url") or ""
