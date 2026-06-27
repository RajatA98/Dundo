"""Slice 1 — build the offline catalog tag sidecar (`catalog_tags.json`) for the Evidence Layer.

Joins MTG-Jamendo editorial tags (genre / mood-theme / instrument) to the served catalog via
`external_ids.jamendoTrackId`. Pod-free, no audio, no MuQ. Catalog tracks lacking a tag-type are
left WITHOUT it (no MuQ backfill — per Codex). Emits a sidecar keyed by track_id plus a `_meta`
block (coarse-genre map, vocab, method), and prints final coverage.

Usage:
    python -m backend.scripts.build_catalog_tags \
        --corpus-rev 3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d \
        --out /tmp/dundo_tags/catalog_tags.json
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

MTG_BASE = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"
MTG_FILES = {
    "genre": "autotagging_genre.tsv",
    "mood": "autotagging_moodtheme.tsv",
    "instrument": "autotagging_instrument.tsv",
}
METHOD = "mtg-knn-v1"

# Coarse super-genre map (display backbone). Unmapped fine genres fall back to themselves.
SUPER_GENRE = {
    "rock": "rock", "hardrock": "rock", "poprock": "rock", "postrock": "rock", "punkrock": "rock",
    "instrumentalrock": "rock", "indie": "rock", "alternative": "rock", "progressive": "rock", "psychedelic": "rock",
    "electronic": "electronic", "edm": "electronic", "house": "electronic", "deephouse": "electronic",
    "techno": "electronic", "trance": "electronic", "dubstep": "electronic", "drumnbass": "electronic",
    "breakbeat": "electronic", "idm": "electronic", "minimal": "electronic", "club": "electronic",
    "dance": "electronic", "electropop": "electronic", "synthpop": "electronic", "eurodance": "electronic",
    "dub": "electronic", "triphop": "electronic", "darkwave": "electronic",
    "pop": "pop", "instrumentalpop": "pop", "popfolk": "pop",
    "classical": "classical", "orchestral": "classical", "symphonic": "classical", "contemporary": "classical",
    "medieval": "classical", "newage": "classical", "opera": "classical", "choir": "classical",
    "jazz": "jazz", "jazzfusion": "jazz", "fusion": "jazz", "swing": "jazz", "bossanova": "jazz",
    "lounge": "jazz", "easylistening": "jazz",
    "folk": "folk", "celtic": "folk", "country": "folk", "ethno": "folk", "world": "folk", "tribal": "folk", "latin": "folk",
    "hiphop": "hiphop", "rap": "hiphop", "rnb": "hiphop", "soul": "hiphop", "funk": "hiphop", "groove": "hiphop",
    "ambient": "ambient", "atmospheric": "ambient", "chillout": "ambient", "experimental": "ambient",
    "improvisation": "ambient", "meditative": "ambient", "soundscape": "ambient",
    "metal": "metal", "hard": "metal", "blues": "blues", "reggae": "reggae", "disco": "disco", "soundtrack": "soundtrack",
}


def _fetch(cache_dir: Path, fname: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / fname
    if not dest.exists():
        url = f"{MTG_BASE}/{fname}"
        print(f"[tags] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def _parse_tsv(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    with open(path) as f:
        next(f, None)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            try:
                tid = int(cols[0].replace("track_", ""))
            except ValueError:
                continue
            tags = [c.split("---")[-1] for c in cols[5:] if "---" in c]
            if tags:
                out[tid] = tags
    return out


def _load_corpus(rev: str, corpus_path: str | None) -> list[dict]:
    if corpus_path:
        return json.loads(Path(corpus_path).read_text())
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("RajatA98/dundo-corpus", "corpus.json", repo_type="dataset", revision=rev)
    return json.loads(Path(p).read_text())


def build(corpus_rev: str, out_path: Path, cache_dir: Path, corpus_path: str | None) -> dict:
    tags = {k: _parse_tsv(_fetch(cache_dir, fname)) for k, fname in MTG_FILES.items()}
    tracks = _load_corpus(corpus_rev, corpus_path)

    catalog: dict[str, dict] = {}
    cov = Counter()
    # Per-label support (how many catalog tracks carry each label) so the backend gate can
    # enforce a minimum-support rule (Codex: rare labels must not become false-precision traps).
    support: dict[str, Counter] = {f: Counter() for f in ("coarseGenre", "genre", "mood", "instrument")}
    for t in tracks:
        tid = t.get("track_id")
        jid_s = (t.get("external_ids") or {}).get("jamendoTrackId")
        if not tid or not (jid_s and jid_s.isdigit()):
            continue
        jid = int(jid_s)
        genre = sorted(set(tags["genre"].get(jid, [])))
        mood = sorted(set(tags["mood"].get(jid, [])))
        instrument = sorted(set(tags["instrument"].get(jid, [])))
        if not (genre or mood or instrument):
            continue  # no MuQ backfill — omit untagged tracks
        entry = {}
        if genre:
            coarse = sorted({SUPER_GENRE.get(g, g) for g in genre})
            entry["genre"] = genre
            entry["coarseGenre"] = coarse
            cov["genre"] += 1
            for g in genre:
                support["genre"][g] += 1
            for g in coarse:
                support["coarseGenre"][g] += 1
        if mood:
            entry["mood"] = mood
            cov["mood"] += 1
            for m in mood:
                support["mood"][m] += 1
        if instrument:
            entry["instrument"] = instrument
            cov["instrument"] += 1
            for i in instrument:
                support["instrument"][i] += 1
        catalog[tid] = entry
        cov["any"] += 1

    out = {
        "_meta": {
            "method": METHOD,
            "source": "MTG-Jamendo autotagging (editorial)",
            "corpusRevision": corpus_rev,
            "trackCount": len(catalog),
            "superGenreMap": SUPER_GENRE,
            "support": {f: dict(c) for f, c in support.items()},
        },
        "tracks": catalog,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))

    n = len(tracks)
    print(f"\n=== catalog_tags.json built: {out_path} ({out_path.stat().st_size/1e6:.1f} MB) ===")
    print(f"catalog tracks: {n}")
    for k in ("genre", "mood", "instrument", "any"):
        print(f"  {k:11s}: {cov[k]:6d} ({100*cov[k]/n:.1f}%)")
    return out


def rehash_manifest_with_sidecar(base_manifest: dict, sidecar_path: Path) -> dict:
    """Add the sidecar to the corpus manifest's sha256 file list + recompute the combined hash,
    so the served dataset's integrity check (validate_manifest_file_hashes) covers catalog_tags.json.
    Mirrors the combined-hash logic in phase2_catalog.py.
    """
    import hashlib
    from backend.scripts.phase2_catalog import compute_file_sha256

    sha = base_manifest.setdefault("sha256", {})
    files = dict(sha.get("files") or {})
    files[sidecar_path.name] = compute_file_sha256(sidecar_path)
    combined = hashlib.sha256()
    for name in sorted(files):
        combined.update(name.encode("utf-8"))
        combined.update(files[name].encode("ascii"))
    sha["files"] = files
    sha["combined"] = combined.hexdigest()
    return base_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-rev", default="3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d")
    ap.add_argument("--corpus-path", default=None, help="local corpus.json (else fetch from HF)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/dundo_tags/catalog_tags.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/mtg_tags_cache"))
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="fetch the corpus manifest and write a rehashed copy including catalog_tags.json")
    a = ap.parse_args()
    build(a.corpus_rev, a.out, a.cache_dir, a.corpus_path)
    if a.manifest_out:
        from huggingface_hub import hf_hub_download
        mpath = hf_hub_download("RajatA98/dundo-corpus", "manifest.json", repo_type="dataset", revision=a.corpus_rev)
        manifest = rehash_manifest_with_sidecar(json.loads(Path(mpath).read_text()), a.out)
        a.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        a.manifest_out.write_text(json.dumps(manifest, separators=(",", ":")))
        print(f"rehashed manifest -> {a.manifest_out} (added {a.out.name}, {len(manifest['sha256']['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
