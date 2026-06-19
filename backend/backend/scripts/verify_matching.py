"""Self-retrieval verification for the Dundo matching pipeline.

For each iTunes Tier-1 track in the catalog: download the actual Apple
preview audio that was used to build the catalog embedding, POST it back
to /neighbors as a fresh query, and check that the same track is returned
at rank 1 with high cosine similarity.

This is the falsifiable answer to "are the matches accurate?" — if the
self-retrieval rate is high, the encoder + retrieval pipeline are doing
what they claim. If it isn't, there's a bug worth finding before we
trust any cross-track match.

Usage:
    # Against the live HF Space (default):
    python -m backend.scripts.verify_matching

    # Against a local backend:
    python -m backend.scripts.verify_matching --base-url http://localhost:8000

    # Just one target:
    python -m backend.scripts.verify_matching --target tier1:itunes:1488408568

ADR-0002 §"Verification" documents the methodology this harness implements.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = REPO_ROOT / "quality-scorer" / "public" / "corpus" / "corpus.json"
DEFAULT_BASE_URL = "https://rajata98-dundo.hf.space"
APPLE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


def main() -> int:
    args = _parse_args()
    catalog = _load_catalog(args.corpus)
    targets = _select_targets(catalog, args.target)
    if not targets:
        print("[verify_matching] no Tier-1 iTunes targets found in catalog; aborting", file=sys.stderr)
        return 2
    print(f"[verify_matching] running against {args.base_url}")
    print(f"[verify_matching] {len(targets)} target(s) to verify")
    print()

    results: list[dict] = []
    for i, target in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] {target['title']} — {target['artist']}")
        try:
            result = _verify_one(target, args.base_url, args.timeout)
        except Exception as exc:
            print(f"  ERROR: {exc!r}")
            result = {
                "track_id": target["track_id"],
                "title": target["title"],
                "artist": target["artist"],
                "error": str(exc),
                "self_rank": None,
                "self_cosine": None,
                "top1_track_id": None,
                "top1_cosine": None,
                "timestamp": None,
            }
        results.append(result)
        _print_one(result)
        time.sleep(args.sleep)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    _print_summary(results)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\n[verify_matching] full results written to {args.json_out}")

    return 0


def _verify_one(target: dict, base_url: str, timeout: float) -> dict:
    preview_url = (target.get("external_ids") or {}).get("previewUrl")
    if not preview_url:
        raise RuntimeError("no previewUrl in catalog entry")

    # Download the iTunes preview. Apple CDN rejects default Python UA so set Safari.
    req = urllib.request.Request(preview_url, headers={"User-Agent": APPLE_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        audio_bytes = r.read()

    # POST to /neighbors. The temp file's .m4a suffix preserves Apple's AAC-LC
    # format so the backend's audioread fallback picks the right decoder.
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx required: pip install -e 'backend/[ingest]'"
            ) from exc
        with httpx.Client(timeout=timeout) as client:
            with open(tmp.name, "rb") as f:
                resp = client.post(
                    f"{base_url}/neighbors",
                    params={"k": 5},
                    files={"file": (f"{target['track_id']}.m4a", f, "audio/mp4")},
                )
        resp.raise_for_status()
        body = resp.json()

    neighbors = body.get("neighbors") or []
    if not neighbors:
        raise RuntimeError(f"empty neighbors response: {body}")

    # Find self-rank in the returned list.
    self_rank = None
    self_cosine = None
    for j, n in enumerate(neighbors, start=1):
        if n.get("trackId") == target["track_id"]:
            self_rank = j
            self_cosine = float(n.get("rawCosine") or 0)
            self_ts = n.get("matchTimestamp") or {}
            break
    else:
        self_ts = {}

    top1 = neighbors[0]
    return {
        "track_id": target["track_id"],
        "title": target["title"],
        "artist": target["artist"],
        "self_rank": self_rank,
        "self_cosine": self_cosine,
        "self_timestamp": self_ts,
        "top1_track_id": top1.get("trackId"),
        "top1_cosine": float(top1.get("rawCosine") or 0),
        "top1_title": (top1.get("track") or {}).get("title"),
        "top1_artist": (top1.get("track") or {}).get("artist"),
    }


def _print_one(r: dict) -> None:
    if r.get("error"):
        return
    self_rank = r.get("self_rank")
    self_cos = r.get("self_cosine")
    top1_id = r.get("top1_track_id")
    if self_rank == 1:
        marker = "PASS"
    elif self_rank is not None:
        marker = f"SELF AT RANK {self_rank}"
    else:
        marker = "SELF NOT IN TOP-5"
    cos_str = f"{self_cos:.4f}" if self_cos is not None else "—"
    print(f"  -> rank-1 returned: {(top1_id or '')[:40]:<40} cos={r.get('top1_cosine', 0):.4f}")
    print(f"     self-retrieval: {marker}  self-cos={cos_str}")
    ts = r.get("self_timestamp") or {}
    if ts and self_rank is not None:
        print(f"     self-timestamp: query {ts.get('queryStartSec','?')}-{ts.get('queryEndSec','?')}s ↔ catalog {ts.get('catalogStartSec','?')}-{ts.get('catalogEndSec','?')}s")


def _print_summary(results: list[dict]) -> None:
    n = len(results)
    succeeded = [r for r in results if not r.get("error")]
    n_ok = len(succeeded)
    if n_ok == 0:
        print("All targets errored out. Check the base URL + network.")
        return
    self_at_1 = [r for r in succeeded if r.get("self_rank") == 1]
    self_in_top5 = [r for r in succeeded if r.get("self_rank") is not None]
    self_cosines = [r["self_cosine"] for r in succeeded if r.get("self_cosine") is not None]
    ts_align = [r for r in succeeded
                if (r.get("self_timestamp") or {}).get("queryStartSec") == (r.get("self_timestamp") or {}).get("catalogStartSec")
                and r.get("self_rank") is not None]

    print(f"Total targets:           {n}")
    print(f"Successful round-trips:  {n_ok}")
    print(f"Self at rank 1:          {len(self_at_1):>3} / {n_ok}  ({100*len(self_at_1)/n_ok:.0f}%)")
    print(f"Self in top 5:           {len(self_in_top5):>3} / {n_ok}  ({100*len(self_in_top5)/n_ok:.0f}%)")
    if self_cosines:
        mean = sum(self_cosines) / len(self_cosines)
        print(f"Self-match cosine mean:  {mean:.4f}  (n={len(self_cosines)})")
        print(f"Self-match cosine min:   {min(self_cosines):.4f}")
    print(f"Timestamp aligns to self: {len(ts_align):>3} / {n_ok}  ({100*len(ts_align)/n_ok:.0f}%)")
    print()
    print("Pass criteria (ADR-0002 §Verification):")
    print(f"  self-retrieval rate >= 90%        ... {'PASS' if len(self_at_1) / n_ok >= 0.9 else 'FAIL'}")
    print(f"  self-cosine mean >= 0.92          ... {'PASS' if self_cosines and sum(self_cosines)/len(self_cosines) >= 0.92 else 'FAIL'}")
    print(f"  timestamp aligns in >= 8 cases    ... {'PASS' if len(ts_align) >= 8 else 'FAIL (or n<8 successful)'}")


def _load_catalog(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else raw.get("tracks", [])


def _select_targets(catalog: list[dict], target_id: str | None) -> list[dict]:
    tier1 = [t for t in catalog if t.get("source") == "itunes" and (t.get("external_ids") or {}).get("previewUrl")]
    if target_id:
        return [t for t in tier1 if t.get("track_id") == target_id]
    return tier1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   help=f"corpus.json path (default: {DEFAULT_CORPUS})")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"backend base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--target", default=None,
                   help="single track_id to verify (default: all Tier-1 iTunes tracks)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="per-request timeout in seconds (default: 120)")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between requests to be polite (default: 1.0)")
    p.add_argument("--json-out", default=None,
                   help="optional path to write full results as JSON")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
