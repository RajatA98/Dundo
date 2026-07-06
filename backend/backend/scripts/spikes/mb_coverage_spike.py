"""THROWAWAY coverage spike (Feature B gate): do our CC/indie Jamendo artists
have retrievable MusicBrainz facts to ground a knowledge narrative?

Samples N artists from artists.json, queries the MusicBrainz search API
(1 req/s, proper UA), disambiguates by name + area, and for confident matches
does a lookup (inc=tags+genres+url-rels) to measure how RICH the facts are.

Prints a coverage report. Provenance-only spike (see scripts/spikes/README.md),
not maintained or run in CI. Run: backend/.venv/bin/python -m backend.scripts.spikes.mb_coverage_spike
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

ARTISTS = "/tmp/artists.json"
SAMPLE = 50
UA = "DundoCoverageSpike/0.1 (rajat1998@gmail.com)"
MB = "https://musicbrainz.org/ws/2"
RATE_S = 1.15  # MB asks <=1 req/s


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def search_artist(name: str, area: str | None) -> dict | None:
    q = f'artist:"{name}"'
    url = f"{MB}/artist/?query={urllib.parse.quote(q)}&fmt=json&limit=5"
    try:
        data = _get(url)
    except Exception as exc:
        print(f"    ! search error for {name!r}: {exc}")
        return None
    for cand in data.get("artists", []):
        score = int(cand.get("score", 0))
        name_match = _norm(cand.get("name")) == _norm(name)
        # confident = exact name match AND MB score high
        if name_match and score >= 90:
            return cand
    return None


def lookup_facts(mbid: str) -> dict:
    url = f"{MB}/artist/{mbid}?inc=tags+genres+url-rels&fmt=json"
    try:
        return _get(url)
    except Exception as exc:
        print(f"    ! lookup error {mbid}: {exc}")
        return {}


def main() -> None:
    artists = json.load(open(ARTISTS))
    named = [a for a in artists if (a.get("name") or "").strip()]
    stride = max(1, len(named) // SAMPLE)
    sample = named[::stride][:SAMPLE]
    print(f"catalog artists: {len(artists)} | named: {len(named)} | sampling {len(sample)} (stride {stride})\n")

    matched = 0
    rich = {"country": 0, "area": 0, "type": 0, "lifeBegin": 0, "tags": 0, "genres": 0, "urls": 0, "disamb": 0}
    examples = []
    for i, a in enumerate(sample, 1):
        name = a["name"].strip()
        loc = a.get("location") or ""
        cand = search_artist(name, loc)
        time.sleep(RATE_S)
        if not cand:
            print(f"  [{i:2}/{len(sample)}] MISS  {name!r}  (loc={loc!r})")
            continue
        matched += 1
        facts = lookup_facts(cand["id"])
        time.sleep(RATE_S)
        country = facts.get("country") or cand.get("country")
        area = (facts.get("area") or cand.get("area") or {}).get("name") if (facts.get("area") or cand.get("area")) else None
        atype = facts.get("type") or cand.get("type")
        begin = (facts.get("life-span") or cand.get("life-span") or {}).get("begin")
        tags = [t["name"] for t in (facts.get("tags") or []) if t.get("count", 0) > 0]
        genres = [g["name"] for g in (facts.get("genres") or [])]
        urls = [r.get("type") for r in (facts.get("relations") or [])]
        disamb = facts.get("disambiguation") or cand.get("disambiguation")
        if country:
            rich["country"] += 1
        if area:
            rich["area"] += 1
        if atype:
            rich["type"] += 1
        if begin:
            rich["lifeBegin"] += 1
        if tags:
            rich["tags"] += 1
        if genres:
            rich["genres"] += 1
        if urls:
            rich["urls"] += 1
        if disamb:
            rich["disamb"] += 1
        print(f"  [{i:2}/{len(sample)}] HIT   {name!r}  country={country} area={area} type={atype} begin={begin} tags={tags[:4]} genres={genres[:3]}")
        if len(examples) < 6:
            examples.append({"name": name, "country": country, "area": area, "type": atype, "begin": begin, "tags": tags[:5], "genres": genres[:3], "disamb": disamb})

    n = len(sample)
    print("\n" + "=" * 60)
    print(f"COVERAGE SPIKE RESULT  (n={n})")
    print(f"  confident MB match: {matched}/{n} = {matched / n:.0%}")
    print("  fact richness among matches (% of matches with the fact):")
    for k, v in rich.items():
        pct = (v / matched) if matched else 0
        print(f"    {k:10}: {v}/{matched} = {pct:.0%}")
    print("\n  example matched facts:")
    for e in examples:
        print(f"    - {e}")
    print("\nGATE: build Feature B if confident-match >= ~50-60% AND >=2-3 rich facts/match.")


if __name__ == "__main__":
    main()
