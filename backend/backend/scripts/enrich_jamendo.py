"""Patch corpus.json's Jamendo tracks with real metadata from Jamendo's public API.

The MTG-Jamendo research dataset (the ingest source for Tier-2) anonymizes
artist names per academic distribution convention — entries land in corpus.json
with placeholders like:

    title:  "Jamendo 382"
    artist: "artist_000020"
    artwork_url: null
    external_ids: {"jamendoTrackId": "382"}

Jamendo's own public Catalog API (`api.jamendo.com/v3.0/tracks/`) returns the
real track name, real artist name, an MP3 stream URL, and an album cover URL
keyed by the same numeric track ID. This script reconciles them.

Frontend impact: the `audioUrlFor()` helper already reads
`external_ids.jamendoAudioUrl` and `artworkUrlFor()` reads `artwork_url`, so
the React rows light up automatically once corpus.json is patched.

Usage:
    JAMENDO_CLIENT_ID=ba16bbc1 \\
        python -m backend.scripts.enrich_jamendo

Options:
    --corpus <path>       Path to corpus.json (default: quality-scorer/public/corpus/corpus.json)
    --dry-run             Print what would change without writing
    --sleep <seconds>     Per-request sleep (default 0.1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = REPO_ROOT / "quality-scorer" / "public" / "corpus" / "corpus.json"
API_BASE = "https://api.jamendo.com/v3.0/tracks/"


def main() -> int:
    args = _parse_args()
    client_id = os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        print("[enrich_jamendo] ERROR: JAMENDO_CLIENT_ID env var required", file=sys.stderr)
        return 2

    raw = json.loads(args.corpus.read_text())
    tracks = raw if isinstance(raw, list) else raw.get("tracks", [])
    jamendo_tracks = [t for t in tracks if t.get("source") == "jamendo"]
    print(f"[enrich_jamendo] {len(jamendo_tracks)} Jamendo tracks to enrich")

    patched = 0
    failed: list[str] = []
    with httpx.Client(timeout=20.0) as client:
        for i, t in enumerate(jamendo_tracks, start=1):
            jam_id = (t.get("external_ids") or {}).get("jamendoTrackId")
            if not jam_id:
                failed.append(f"{t.get('track_id')}: missing jamendoTrackId")
                continue
            try:
                r = client.get(API_BASE, params={
                    "client_id": client_id,
                    "id": str(jam_id),
                    "format": "json",
                })
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                failed.append(f"jamendo:{jam_id}: {exc}")
                time.sleep(args.sleep)
                continue

            if data.get("headers", {}).get("status") != "success":
                failed.append(f"jamendo:{jam_id}: {data.get('headers', {}).get('error_message', 'unknown')}")
                time.sleep(args.sleep)
                continue

            results = data.get("results") or []
            if not results:
                failed.append(f"jamendo:{jam_id}: not in Jamendo catalog")
                time.sleep(args.sleep)
                continue

            jam = results[0]
            real_title = jam.get("name") or t.get("title")
            real_artist = jam.get("artist_name") or t.get("artist")
            audio_url = jam.get("audio") or None
            image_url = jam.get("image") or None
            track_view = f"https://www.jamendo.com/track/{jam_id}"

            if args.dry_run:
                print(f"  [{i:03d}/{len(jamendo_tracks)}] {t.get('track_id')}: {t.get('title')!r} -> {real_title!r} by {real_artist!r}")
            else:
                t["title"] = real_title
                t["artist"] = real_artist
                t["artwork_url"] = image_url
                t["track_view_url"] = track_view
                ext = dict(t.get("external_ids") or {})
                if audio_url:
                    ext["jamendoAudioUrl"] = audio_url
                if jam.get("album_name"):
                    ext["jamendoAlbum"] = jam.get("album_name")
                t["external_ids"] = ext

            patched += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(jamendo_tracks)}] enriched so far: {patched}, failed: {len(failed)}")
            time.sleep(args.sleep)

    print(f"[enrich_jamendo] DONE: patched={patched} failed={len(failed)}")
    if failed:
        print("[enrich_jamendo] failures (first 10):")
        for f in failed[:10]:
            print(f"  - {f}")

    if not args.dry_run and patched > 0:
        args.corpus.write_text(json.dumps(raw, indent=2))
        print(f"[enrich_jamendo] wrote {args.corpus}")
    elif args.dry_run:
        print("[enrich_jamendo] dry-run — no file written")

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
