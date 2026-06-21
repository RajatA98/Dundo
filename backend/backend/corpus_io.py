"""Read/write the static corpus assets (current 145-track seed path).

Phase-2 note (REVIEW_REPORT.md Minor #3): this is the *legacy* local-file corpus
path. Phase 2 moves the served corpus to the HF Dataset repo (`RajatA98/dundo-corpus`)
loaded into RAM at boot. The corpus *writer* used by the build pipeline is
`scripts/_corpus_writer.py`; this module only handles local read/stats/examples
helpers for the seed. Don't confuse the two when implementing Phase 2 storage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


def compute_stats(tracks: list[dict]) -> dict:
    """Mirror the JS `corpusStats()` shape so the Corpus page can render directly."""
    total = len(tracks)
    dropped = [t for t in tracks if t.get("verdict") == "drop"]
    by_mode: dict[str, int] = {}
    for t in dropped:
        m = t.get("primaryFail") or "other"
        by_mode[m] = by_mode.get(m, 0) + 1
    return {
        "total": total,
        "kept": total - len(dropped),
        "dropped": len(dropped),
        "keepPct": round((total - len(dropped)) / total * 100) if total else 0,
        "byMode": by_mode,
    }


EXAMPLE_PICKS: list[tuple[str, Callable[[dict], bool]]] = [
    ("clean", lambda t: t.get("verdict") == "keep" and t.get("score", 0) >= 80),
    ("clipped", lambda t: t.get("primaryFail") == "clipping"),
    ("silent", lambda t: t.get("primaryFail") == "silence"),
    ("noisy", lambda t: t.get("primaryFail") == "noise"),
    ("truncated", lambda t: t.get("primaryFail") == "truncation"),
]


def pick_examples(tracks: list[dict]) -> list[dict]:
    """Pick 5 curated archetypes for the Scorer page sidebar (one per failure mode + a clean keep)."""
    out: list[dict] = []
    for label, predicate in EXAMPLE_PICKS:
        match = next((t for t in tracks if predicate(t)), None)
        if match is None:
            continue
        ex = dict(match)
        ex["example"] = label
        ex["chipLabel"] = label.capitalize()
        ex["source"] = "example"
        out.append(ex)
    return out


def write_corpus(
    out_dir: Path,
    tracks: list[dict],
    embeddings: np.ndarray,
    *,
    version: str = "0.1.0",
) -> dict:
    """Write `corpus.json`, `embeddings.npy`, `corpus_stats.json`, `examples.json`.

    Returns the stats dict (handy for printing a summary at end of ingest).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "tracks": tracks,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "corpus.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )

    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))

    stats = compute_stats(tracks)
    (out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2))

    examples = pick_examples(tracks)
    (out_dir / "examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False)
    )

    return stats


def read_existing_track_ids(out_dir: Path) -> set[str]:
    """For resumability: ids already written to corpus.json (empty set if none)."""
    p = Path(out_dir) / "corpus.json"
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return {t["id"] for t in data.get("tracks", [])}
    except Exception:
        return set()
