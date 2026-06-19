"""SONICS subsample downloader.

The dataset (`awsaf49/sonics`) stores its synthetic-half mp3s inside **10 zip
archives** at `fake_songs/part_NN.zip` — each ~3.5 GB. The metadata index and
the zip contents are NOT aligned by row order (metadata streams from id ~54000s
while early zips hold lower ids), so naively picking from metadata then matching
against a zip yields ~0 hits.

This module inverts the order:
  1. Download the requested zip `parts` (cached in `~/.cache/huggingface/`).
  2. Read each zip's mp3 filenames (the **ground truth** of what's available).
  3. Filter to Suno/Udio rows by filename pattern (`fake_NNNN_{suno|udio}_M.mp3`).
  4. Random-sample `n` filenames across the available pool.
  5. Extract just those mp3s into `out_dir`.
  6. (Optional) attach SONICS metadata (genre/mood/style) per track — left
     off by default because the dataset stream is slow; track titles fall
     back to the filename stem, which is informative on its own.

For a smoke path (no download), pass `n=0`.
"""

from __future__ import annotations

import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "awsaf49/sonics"
SOURCES = ("suno", "udio")             # the synthetic half we want
PART_ZIP_TEMPLATE = "fake_songs/part_{:02d}.zip"
N_PARTS = 10
ATTRIBUTION_URL = f"https://huggingface.co/datasets/{REPO_ID}"

_SOURCE_FROM_NAME = re.compile(r"_(suno|udio)_")


def _classify(filename: str) -> str | None:
    m = _SOURCE_FROM_NAME.search(filename)
    return m.group(1) if m else None


def _ensure_part_cached(part_n: int) -> tuple[str, list[str]]:
    """Download part_NN.zip (HF cache) and return (zip_path, suno/udio mp3 names)."""
    zname = PART_ZIP_TEMPLATE.format(part_n)
    print(f"[sonics] fetching {zname} (~3.5 GB on first run, cached after)…")
    zpath = hf_hub_download(repo_id=REPO_ID, filename=zname, repo_type="dataset")
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".mp3") and _classify(n)]
    print(f"[sonics] part_{part_n:02d}: {len(names)} Suno/Udio mp3s available")
    return zpath, names


def _stratified_pick(
    candidates: list[tuple[int, str]], n: int, rng: random.Random
) -> list[tuple[int, str]]:
    """Round-robin draw across source (suno/udio) buckets for balance."""
    bucketed: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for part_n, fname in candidates:
        bucketed[_classify(fname) or "?"].append((part_n, fname))
    for v in bucketed.values():
        rng.shuffle(v)
    iters = {k: iter(v) for k, v in bucketed.items()}
    picks: list[tuple[int, str]] = []
    while len(picks) < n and iters:
        for k in list(iters):
            try:
                picks.append(next(iters[k]))
                if len(picks) >= n:
                    break
            except StopIteration:
                del iters[k]
    return picks


def download_subsample(
    n: int,
    out_dir: Path,
    seed: int = 42,
    parts: tuple[int, ...] = (1,),
    progress: bool = True,
) -> list[dict]:
    """Pick + extract `n` Suno/Udio tracks from the requested zip `parts`.

    Returns descriptors with the local extracted audio path + attribution stub.
    Each requested part covers ~10% of the dataset; for full ~300-track coverage,
    request all 10 parts (~35 GB).
    """
    if n <= 0:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 1. Download (or cache-hit) every requested zip; collect available mp3s.
    zip_paths: dict[int, str] = {}
    candidates: list[tuple[int, str]] = []
    for p in parts:
        zpath, names = _ensure_part_cached(p)
        zip_paths[p] = zpath
        for fname in names:
            candidates.append((p, fname))

    if not candidates:
        raise RuntimeError(
            "No Suno/Udio mp3s in the requested parts — check connectivity / parts spec"
        )

    # 2. Sample N filenames stratified across source.
    picks = _stratified_pick(candidates, n, rng)
    print(f"[sonics] picked {len(picks)} tracks across {len({_classify(f) for _, f in picks})} sources")

    # 3. Extract picks from their respective zips.
    by_part: dict[int, list[str]] = defaultdict(list)
    for p, fname in picks:
        by_part[p].append(fname)

    descriptors: list[dict] = []
    for part_n, fnames in by_part.items():
        with zipfile.ZipFile(zip_paths[part_n]) as zf:
            it: Iterable[str] = fnames
            if progress:
                it = tqdm(fnames, desc=f"Extract part_{part_n:02d}", unit="track")
            for fname in it:
                extracted_at = Path(zf.extract(fname, out_dir))
                source_kind = _classify(fname) or "unknown"
                # Filename pattern: fake_<id>_{suno|udio}_<v>.mp3
                stem = Path(fname).stem
                source_id = stem.split("_")[1] if "_" in stem else stem
                descriptors.append({
                    "id": f"sonics-{source_kind}-{source_id}",
                    "source": "sonics",
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "local_path": str(extracted_at),
                    "attribution": {
                        "source": "sonics",
                        "license": "CC BY-NC 4.0",
                        "url": ATTRIBUTION_URL,
                    },
                })
    return descriptors
