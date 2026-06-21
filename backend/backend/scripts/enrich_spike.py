"""Enrichment-spike harness (Phase 1) — MEASURE coverage before scaling.

The point of this script (Codex review ⭐#6): do NOT commit the enrichment pipeline
or the 160K encode until we have *measured*, on a representative sample, how much of
each artist-card field we can actually populate. Listen link should be ~100%
(guaranteed host page); location is expected ~40-55%; rich external support links and
Spotify will be lower.

Division of labor:
  * Claude scaffolded: the provider Protocol, the sample selection, the merge logic,
    and the coverage report. This file RUNS today and reports ~100% listenUrl + 0%
    everything-else (because the providers are stubs).
  * Codex implements the three provider bodies marked ``# CODEX:`` — JamendoEnricher,
    FMAEnricher, MusicBrainzEnricher — per
    ``factory/artifacts/CODEX_PHASE_1_IMPLEMENTATION.md``.

Run:
    python -m backend.scripts.enrich_spike --sample 1000
Writes:
    factory/artifacts/ENRICHMENT_COVERAGE.json   (machine-readable coverage report)
and prints a human summary. The go/no-go on scaling the pipeline is read off this.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import urlparse

from backend.artist import ArtistRecord, SupportLink, aggregate_tracks_by_artist

try:
    import httpx
except ImportError:  # pragma: no cover - ingest extra normally provides this.
    httpx = None

JAMENDO_API_BASE = "https://api.jamendo.com/v3.0"


# --------------------------------------------------------------------------------
# Provider interface + results
# --------------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """What a provider can add for one artist. All fields optional/best-effort."""

    location: Optional[str] = None  # "City, Country" (normalize FMA free-text; Jamendo is ISO)
    supportLinks: list[SupportLink] = field(default_factory=list)
    spotifyUrl: Optional[str] = None
    # The corroborating signal that justified spotifyUrl (MBID / homepage / bandcamp
    # cross-link / source id). MUST be non-None whenever spotifyUrl is set, else the
    # merge step drops the Spotify link (PRESEARCH / Codex #5 confidence gate).
    spotifyConfidence: Optional[str] = None
    sourceArtistId: Optional[str] = None  # real source artist id, when resolved


class ArtistEnricher(Protocol):
    name: str

    def enrich(self, artist: ArtistRecord) -> EnrichmentResult:
        ...


# --------------------------------------------------------------------------------
# Providers — CODEX implements the bodies. Stubs return empty so the harness runs.
# --------------------------------------------------------------------------------


class JamendoEnricher:
    """Jamendo API: location (artists/locations -> ISO country+city) + website/shareurl."""

    name = "jamendo"

    def __init__(
        self,
        client_id: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        sleep_seconds: float = 0.1,
        client: Any = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("JAMENDO_CLIENT_ID")
        self.cache_dir = cache_dir or _default_cache_dir() / "jamendo"
        self.sleep_seconds = sleep_seconds
        self.client = client

    def enrich(self, artist: ArtistRecord) -> EnrichmentResult:
        if artist.source != "jamendo":
            return EnrichmentResult()
        if not (self.client_id or self.client):
            return EnrichmentResult()

        source_artist_id = artist.sourceArtistId or self._resolve_artist_id(artist)
        if not source_artist_id:
            return EnrichmentResult()

        info = self._get("artists", {"id": source_artist_id}) or {}
        loc = self._get("artists/locations", {"id": source_artist_id}) or {}
        artist_info = _first_result(info)
        # The artists/locations endpoint nests the location under results[0].locations[],
        # not directly on results[0]. Pull the first declared location.
        location_info = _first_result(loc)
        loc_list = location_info.get("locations") or []
        first_loc = loc_list[0] if loc_list and isinstance(loc_list[0], dict) else {}

        links: list[SupportLink] = []
        share_url = _clean_url(artist_info.get("shareurl"))
        if share_url:
            links.append(SupportLink(kind="jamendo", url=share_url, label="Jamendo"))
        website = _clean_url(artist_info.get("website"))
        if website:
            links.append(SupportLink(kind="website", url=website, label="Website"))

        return EnrichmentResult(
            sourceArtistId=str(source_artist_id),
            location=_format_location(first_loc.get("city"), first_loc.get("country")),
            supportLinks=links,
        )

    def _resolve_artist_id(self, artist: ArtistRecord) -> Optional[str]:
        data = self._get("artists", {"namesearch": artist.name, "limit": "1"})
        result = _first_result(data or {})
        artist_id = result.get("id")
        return str(artist_id) if artist_id not in (None, "") else None

    def _get(self, endpoint: str, params: dict[str, Any]) -> Optional[dict]:
        request_params = {"format": "json", **params}
        if self.client_id:
            request_params["client_id"] = self.client_id
        cache_key = _cache_key(endpoint, request_params)
        cached = _read_json(self.cache_dir / f"{cache_key}.json")
        if cached is not None:
            return cached

        client = self.client
        close_client = False
        if client is None:
            if httpx is None:
                return None
            client = httpx.Client(timeout=20.0)
            close_client = True
        try:
            response = client.get(f"{JAMENDO_API_BASE}/{endpoint.strip('/')}/", params=request_params)
            response.raise_for_status()
            data = response.json()
        finally:
            if close_client:
                client.close()
        if data.get("headers", {}).get("status") not in (None, "success"):
            return None
        _write_json(self.cache_dir / f"{cache_key}.json", data)
        if self.sleep_seconds > 0 and self.client is None:
            time.sleep(self.sleep_seconds)
        return data


class FMAEnricher:
    """FMA metadata dump: artist_website/url/donation + artist_location (free-text)."""

    name = "fma"

    def __init__(self, fma_metadata_dir: Optional[Path] = None) -> None:
        self.fma_metadata_dir = fma_metadata_dir
        self._by_artist: Optional[dict[str, dict[str, str]]] = None

    def enrich(self, artist: ArtistRecord) -> EnrichmentResult:
        if artist.source != "fma":
            return EnrichmentResult()
        row = self._artist_rows().get(_norm_name(artist.name))
        if not row:
            return EnrichmentResult()

        links: list[SupportLink] = []
        website = _first_value(row, "artist_website", "website", "artist_site")
        if website:
            links.append(SupportLink(kind="website", url=website, label="Website"))
        fma_url = _first_value(row, "artist_url", "url", "artist_page")
        if fma_url:
            links.append(SupportLink(kind="fma", url=fma_url, label="Free Music Archive"))
        donation = _first_value(row, "artist_donation_url", "donation_url", "artist_donation")
        if donation:
            links.append(SupportLink(kind=_support_kind_for_url(donation), url=donation, label=_label_for_url(donation)))

        return EnrichmentResult(
            location=_first_value(row, "artist_location", "location"),
            supportLinks=links,
            sourceArtistId=_first_value(row, "artist_id", "id"),
        )

    def _artist_rows(self) -> dict[str, dict[str, str]]:
        if self._by_artist is not None:
            return self._by_artist
        self._by_artist = {}
        if not self.fma_metadata_dir:
            return self._by_artist
        paths = [self.fma_metadata_dir] if self.fma_metadata_dir.is_file() else sorted(self.fma_metadata_dir.glob("*.csv"))
        for path in paths:
            try:
                with path.open(newline="") as f:
                    for row in csv.DictReader(f):
                        normalized = {_norm_key(k): (v or "").strip() for k, v in row.items() if k}
                        name = _first_value(normalized, "artist_name", "artist", "artist_title", "name")
                        if name:
                            self._by_artist.setdefault(_norm_name(name), normalized)
            except OSError:
                continue
        return self._by_artist


class MusicBrainzEnricher:
    """MusicBrainz supplement: area (location) + url-rels (Bandcamp/homepage/streaming).

    Also the source of the Spotify confidence signal: only surface a Spotify URL when
    MusicBrainz (or a homepage/Bandcamp cross-link) corroborates the identity.
    """

    name = "musicbrainz"

    def __init__(self, dump_path: Optional[Path] = None) -> None:
        self.dump_path = dump_path
        self._by_artist: Optional[dict[str, dict[str, Any]]] = None

    def enrich(self, artist: ArtistRecord) -> EnrichmentResult:
        record = self._artist_rows().get(_norm_name(artist.name))
        if not record:
            return EnrichmentResult()

        links: list[SupportLink] = []
        spotify_url = None
        corroborating_link = False
        for rel in _iter_mb_urls(record):
            url = _clean_url(rel.get("url"))
            if not url:
                continue
            kind = _support_kind_for_url(url, rel.get("type"))
            if kind == "spotify":
                spotify_url = url
                continue
            if kind in {"bandcamp", "website", "patreon"}:
                corroborating_link = True
                links.append(SupportLink(kind=kind, url=url, label=_label_for_url(url, rel.get("type"))))

        spotify_confidence = None
        if spotify_url and record.get("mbid") and corroborating_link:
            spotify_confidence = "mbid+external-link"
        else:
            spotify_url = None

        return EnrichmentResult(
            location=_musicbrainz_location(record),
            supportLinks=links,
            spotifyUrl=spotify_url,
            spotifyConfidence=spotify_confidence,
            sourceArtistId=str(record.get("mbid")) if record.get("mbid") else None,
        )

    def _artist_rows(self) -> dict[str, dict[str, Any]]:
        if self._by_artist is not None:
            return self._by_artist
        self._by_artist = {}
        if not self.dump_path or not self.dump_path.exists():
            return self._by_artist
        data = _read_json(self.dump_path)
        rows = data.get("artists", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return self._by_artist
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("name") or row.get("artist") or row.get("sort-name")
            if name:
                self._by_artist.setdefault(_norm_name(str(name)), row)
        return self._by_artist


def default_enrichers() -> list[ArtistEnricher]:
    fma_dir = os.environ.get("FMA_METADATA_DIR")
    mb_dump = os.environ.get("MUSICBRAINZ_ARTIST_DUMP")
    return [
        JamendoEnricher(),
        FMAEnricher(Path(fma_dir) if fma_dir else None),
        MusicBrainzEnricher(Path(mb_dump) if mb_dump else None),
    ]


# --------------------------------------------------------------------------------
# Merge + measurement (Claude-implemented; do not change the metric definitions
# without updating the acceptance gate in PROJECT_PLAN.md / the Codex instruction).
# --------------------------------------------------------------------------------


def merge(artist: ArtistRecord, results: list[EnrichmentResult]) -> ArtistRecord:
    """Fold provider results into the artist record. Provider order = priority."""
    for r in results:
        if r.sourceArtistId and not artist.sourceArtistId:
            artist.sourceArtistId = r.sourceArtistId
        if r.location and not artist.location:
            artist.location = r.location
        for link in r.supportLinks:
            if all(existing.url != link.url for existing in artist.supportLinks):
                artist.supportLinks.append(link)
        # Confidence gate: only accept a Spotify link with a corroborating signal.
        if r.spotifyUrl and r.spotifyConfidence and not artist.spotifyUrl:
            artist.spotifyUrl = r.spotifyUrl
    return artist


def _sample(records: dict[str, ArtistRecord], n: int) -> list[ArtistRecord]:
    """Deterministic, source-stratified sample (reproducible across runs)."""
    by_source: dict[str, list[ArtistRecord]] = {}
    for rec in records.values():
        by_source.setdefault(rec.source, []).append(rec)
    picked: list[ArtistRecord] = []
    for source, recs in sorted(by_source.items()):
        recs.sort(key=lambda r: r.artistId)
        quota = max(1, round(n * len(recs) / len(records)))
        if len(recs) <= quota:
            picked.extend(recs)
        else:
            stride = len(recs) / quota
            picked.extend(recs[int(i * stride)] for i in range(quota))
    return picked


def run_spike(
    tracks: list[dict], sample_size: int, enrichers: list[ArtistEnricher]
) -> dict:
    """Aggregate -> sample -> enrich -> measure coverage. Returns the report dict."""
    records = aggregate_tracks_by_artist(tracks)
    sample = _sample(records, sample_size)

    for rec in sample:
        merge(rec, [e.enrich(rec) for e in enrichers])

    n = len(sample) or 1
    have_listen = sum(1 for r in sample if r.listenUrl)
    have_location = sum(1 for r in sample if r.location)
    have_rich_support = sum(
        1 for r in sample if any(l.kind in ("bandcamp", "patreon", "website") for l in r.supportLinks)
    )
    have_spotify = sum(1 for r in sample if r.spotifyUrl)

    def pct(x: int) -> float:
        return round(100 * x / n, 1)

    return {
        "totalArtists": len(records),
        "sampleSize": len(sample),
        "bySource": {
            s: sum(1 for r in records.values() if r.source == s)
            for s in sorted({r.source for r in records.values()})
        },
        "coveragePct": {
            "listenUrl": pct(have_listen),       # ACCEPTANCE: must be ~100%
            "location": pct(have_location),       # expected ~40-55%
            "richSupportLink": pct(have_rich_support),
            "spotify": pct(have_spotify),
        },
        "notes": _coverage_note(sample),
    }


def _coverage_note(sample: list[ArtistRecord]) -> str:
    if not sample:
        return "No artists sampled."
    if not any(r.location or r.supportLinks or r.spotifyUrl for r in sample):
        return (
            "Providers are implemented, but no enrichment sources were configured for this run. "
            "Set JAMENDO_CLIENT_ID, FMA_METADATA_DIR, or MUSICBRAINZ_ARTIST_DUMP to measure real coverage."
        )
    fields = ["listenUrl"]
    if any(r.location for r in sample):
        fields.append("location")
    if any(r.supportLinks for r in sample):
        fields.append("supportLinks")
    if any(r.spotifyUrl for r in sample):
        fields.append("spotifyUrl")
    return f"Renderable fields with observed data in this sample: {', '.join(fields)}."


def _default_cache_dir() -> Path:
    return Path(os.environ.get("DUNDO_ENRICH_CACHE", ".cache/dundo-enrich"))


def _cache_key(endpoint: str, params: dict[str, Any]) -> str:
    raw = endpoint + "__" + "__".join(f"{k}={params[k]}" for k in sorted(params) if k != "client_id")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("_")


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def _first_result(data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results") or []
    return results[0] if results and isinstance(results[0], dict) else {}


def _format_location(city: Any, country: Any) -> Optional[str]:
    parts = [str(p).strip() for p in (city, country) if str(p or "").strip()]
    return ", ".join(parts) if parts else None


def _clean_url(url: Any) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    if url.startswith(("http://", "https://")):
        return url
    if "." in url and " " not in url:
        return f"https://{url}"
    return None


def _norm_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _norm_name(name: str) -> str:
    return re.sub(r"[\W_]+", " ", name.casefold()).strip()


def _first_value(row: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = row.get(_norm_key(key), row.get(key))
        if isinstance(value, str):
            value = value.strip()
        if value not in (None, ""):
            return str(value)
    return None


def _support_kind_for_url(url: str, rel_type: Any = None) -> str:
    rel = str(rel_type or "").casefold()
    host = urlparse(url).netloc.casefold()
    text = f"{rel} {host}"
    if "spotify" in text:
        return "spotify"
    if "bandcamp" in text:
        return "bandcamp"
    if "patreon" in text:
        return "patreon"
    if "freemusicarchive" in text:
        return "fma"
    if "jamendo" in text:
        return "jamendo"
    if "homepage" in rel or "official" in rel or "website" in rel:
        return "website"
    return "website"


def _label_for_url(url: str, rel_type: Any = None) -> str:
    kind = _support_kind_for_url(url, rel_type)
    return {
        "bandcamp": "Bandcamp",
        "patreon": "Patreon",
        "fma": "Free Music Archive",
        "jamendo": "Jamendo",
        "spotify": "Spotify",
        "website": "Website",
    }.get(kind, "Website")


def _iter_mb_urls(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("urls") or record.get("url-rels") or record.get("relations") or []
    urls: list[dict[str, str]] = []
    if isinstance(raw, dict):
        raw = [{"type": k, "url": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        return urls
    for item in raw:
        if isinstance(item, str):
            urls.append({"type": "", "url": item})
        elif isinstance(item, dict):
            target = item.get("url")
            if isinstance(target, dict):
                target = target.get("resource")
            urls.append({"type": str(item.get("type") or item.get("relation-type") or ""), "url": str(target or "")})
    return urls


def _musicbrainz_location(record: dict[str, Any]) -> Optional[str]:
    area = record.get("area") or record.get("begin-area")
    if isinstance(area, dict):
        return _first_value(area, "name", "sort-name")
    if isinstance(area, str):
        return area.strip() or None
    return None


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def _find_corpus_json() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "quality-scorer" / "public" / "corpus" / "corpus.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate quality-scorer/public/corpus/corpus.json")


def _report_path() -> Path:
    corpus = _find_corpus_json()
    repo_root = corpus.parents[3]  # .../<repo>/quality-scorer/public/corpus/corpus.json
    return repo_root / "factory" / "artifacts" / "ENRICHMENT_COVERAGE.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dundo Phase-1 enrichment coverage spike")
    parser.add_argument("--sample", type=int, default=1000, help="target sample size")
    parser.add_argument("--corpus", type=Path, default=None, help="path to corpus.json")
    args = parser.parse_args()

    corpus_path = args.corpus or _find_corpus_json()
    tracks = json.loads(corpus_path.read_text())

    report = run_spike(tracks, args.sample, default_enrichers())

    out = _report_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    cov = report["coveragePct"]
    print(f"Artists: {report['totalArtists']}  sample: {report['sampleSize']}  by source: {report['bySource']}")
    print("Coverage %:")
    for k, v in cov.items():
        print(f"  {k:18s} {v}")
    print(f"\nReport written to {out}")
    if cov["listenUrl"] < 95:
        print("WARN: listenUrl coverage below 95% — the guaranteed link is not actually guaranteed.")


if __name__ == "__main__":
    main()
