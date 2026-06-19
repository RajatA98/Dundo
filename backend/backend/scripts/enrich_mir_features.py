"""Backfill the four ADR-0004 MIR features (tempo, key, chroma, MFCC) into
corpus.json without re-encoding the catalog through MuQ-MuLan.

Pattern matches `enrich_jamendo.py`: read corpus.json, for each entry that
lacks `mir_features`, download its audio, run `mir_features.compute()`,
write the result back. Idempotent — re-runs only touch un-enriched tracks.

Usage:
    python -m backend.scripts.enrich_mir_features

The audio source depends on the tier:
    - tier1 (iTunes):  external_ids.previewUrl — 30s AAC-LC preview
    - tier2 (Jamendo): external_ids.jamendoAudioUrl — MP3 stream (set by
                      enrich_jamendo earlier; falls back to source_url if missing)

Cost: ~3 s per track on CPU (download + librosa decode + features). For the
current 155-track catalog: ~8 min wall-clock.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import librosa

from backend.mir_features import compute as compute_mir

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = REPO_ROOT / "quality-scorer" / "public" / "corpus" / "corpus.json"
APPLE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 Safari/605.1.15"


def main() -> int:
    args = _parse_args()
    raw = json.loads(args.corpus.read_text())
    tracks = raw if isinstance(raw, list) else raw.get("tracks", [])
    total = len(tracks)
    print(f"[enrich_mir] {total} catalog tracks")

    todo = [t for t in tracks if args.force or not t.get("mir_features")]
    print(f"[enrich_mir] {len(todo)} to enrich (already-enriched skipped; pass --force to redo)")

    patched = 0
    failed: list[str] = []
    for i, t in enumerate(todo, start=1):
        track_id = t.get("track_id", "?")
        try:
            url = _audio_url_for(t)
            if not url:
                failed.append(f"{track_id}: no audio URL in corpus entry")
                continue
            audio_bytes = _download(url, args.timeout)
            wav, sr = _decode(audio_bytes)
            features = compute_mir(wav, sr)
            t["mir_features"] = features.to_dict()
            patched += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i:>3}/{len(todo)}] enriched={patched} failed={len(failed)}")
        except Exception as exc:
            failed.append(f"{track_id}: {exc!r}")
        time.sleep(args.sleep)

    args.corpus.write_text(json.dumps(raw, indent=2))
    print(f"[enrich_mir] DONE: patched={patched} failed={len(failed)}")
    if failed:
        print("[enrich_mir] first 10 failures:")
        for f in failed[:10]:
            print(f"  - {f}")
    return 0 if patched > 0 or not todo else 1


def _audio_url_for(track: dict) -> str | None:
    """Pick the best audio source URL for a track per its tier."""
    ext = track.get("external_ids") or {}
    return (
        ext.get("previewUrl")            # iTunes Tier-1
        or ext.get("jamendoAudioUrl")    # Jamendo enriched
        or ext.get("jamendoStreamUrl")
        or track.get("source_url")       # last resort
    )


def _download(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": APPLE_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _decode(audio_bytes: bytes):
    """Decode arbitrary audio bytes to a mono numpy array via the temp-file
    path so AAC-LC (.m4a) works the same way it does in api.py.
    """
    # Try BytesIO first (works for mp3/wav/flac/ogg).
    try:
        wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
        if wav.size > 0:
            return wav, sr
    except Exception:
        pass
    # Fall through to temp-file path.
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav, sr = librosa.load(tmp.name, sr=22050, mono=True)
    return wav, sr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--force", action="store_true",
                   help="Re-enrich even tracks that already have mir_features")
    p.add_argument("--sleep", type=float, default=0.2,
                   help="Per-track sleep to be polite to source CDNs")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-request download timeout")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
