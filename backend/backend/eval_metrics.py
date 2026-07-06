"""Eval metrics that check cross-artist relevance using real catalog tags.

`run_leave_one_out` (run_eval.py) measures self-retrieval — a track finding
another track by the same artist. It answers "can the pipeline find a track
at all?" It does NOT answer the product claim: "does a *different* artist's
match genuinely resemble the upload?" `cross_artist_tag_overlap` answers
that second question, grounded in real MTG-Jamendo editorial tags (no LLM
judge, no new manual labeling) — consistent with this project's
heuristics-first eval philosophy (ADR-0005).
"""
from __future__ import annotations

import random

# Scoped to genre + mood only. Instrument overlap ("both tracks have guitar")
# is too generic a signal to count as evidence two tracks genuinely resemble
# each other — most of the catalog has a guitar.
_TAG_FIELDS = ("coarseGenre", "mood")


def _real_tags(entry: dict | None) -> set[tuple[str, str]]:
    if not entry:
        return set()
    out: set[tuple[str, str]] = set()
    for field in _TAG_FIELDS:
        for label in entry.get(field) or []:
            out.add((field, label))
    return out


def cross_artist_tag_overlap(
    loo_results: list[dict],
    catalog_tags: dict[str, dict],
    *,
    top_k: int = 3,
    random_seed: int = 42,
) -> dict:
    """For each leave-one-out query with real tags, check whether its top-k
    *genuinely different-artist* neighbors (excluding `target_track_ids`,
    the query's own same-artist tracks) share a real genre/mood tag with the
    query — and compare that rate against a random-baseline floor (the same
    check against `top_k` randomly sampled different-artist catalog tracks).

    Returns:
        {
          "n": number of queries actually evaluated (has real tags AND at
              least one genuinely-different-artist neighbor),
          "skippedNoTags": queries skipped because the query track itself
              has no real tags in catalog_tags,
          "crossArtistTagOverlapAt3": fraction of evaluated queries where at
              least one cross-artist top-k neighbor shared a tag, or None
              if n == 0,
          "randomBaselineTagOverlapAt3": same check against random
              different-artist tracks instead of the actual top-k, or None
              if n == 0 — read the first number against this floor.
        }
    """
    rng = random.Random(random_seed)
    all_track_ids = list(catalog_tags.keys())
    matched = 0
    matched_random = 0
    evaluated = 0
    skipped_no_tags = 0

    for row in loo_results:
        query_id = row["query_track_id"]
        query_tags = _real_tags(catalog_tags.get(query_id))
        if not query_tags:
            skipped_no_tags += 1
            continue

        same_artist = set(row.get("target_track_ids") or [])
        cross_artist_neighbors = [
            n
            for n in row["top_neighbors"]
            if n["trackId"] != query_id and n["trackId"] not in same_artist
        ][:top_k]
        if not cross_artist_neighbors:
            continue

        evaluated += 1
        if any(
            _real_tags(catalog_tags.get(n["trackId"])) & query_tags
            for n in cross_artist_neighbors
        ):
            matched += 1

        candidate_pool = [
            t for t in all_track_ids if t != query_id and t not in same_artist
        ]
        random_picks = rng.sample(candidate_pool, min(top_k, len(candidate_pool)))
        if any(_real_tags(catalog_tags.get(t)) & query_tags for t in random_picks):
            matched_random += 1

    return {
        "n": evaluated,
        "skippedNoTags": skipped_no_tags,
        "crossArtistTagOverlapAt3": round(matched / evaluated, 4) if evaluated else None,
        "randomBaselineTagOverlapAt3": round(matched_random / evaluated, 4)
        if evaluated
        else None,
    }
