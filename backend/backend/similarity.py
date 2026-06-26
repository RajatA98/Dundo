"""Catalog similarity math — mean-pooled + max-segment cosine ranking.

Phase 2 module. Owns the math the `/neighbors` endpoint uses to rank catalog
tracks against a query. Kept separate from `api.py` for testability and to keep
the FastAPI handler short.

Contract (LOCKED_DECISIONS + PROJECT_PLAN Phase 2):

- Catalog embeddings are L2-normalized; query embeddings (mean + per-window
  segments) are L2-normalized.
- `meanPooledSimilarity` per neighbor = dot(query_mean, catalog_mean). This is
  the primary ranking signal (matches the headline percentage shown in the UI).
- `maxSegmentSimilarity` per neighbor = max over (i, j) of
  dot(query_segment_i, catalog_segment_j). Reveals local resemblance even when
  the mean-pooled similarity washes out (e.g. a one-bar melodic match).
- Top-k is sorted by meanPooledSimilarity descending. maxSegmentSimilarity is
  surfaced alongside as a secondary metric, never as the rank key.

Performance note: at N≈500 tracks × ~3 segments each, the cross-segment matrix
is ~9 × 1500 × 512 ≈ 7 M dot-product ops. Sub-millisecond on CPU. We bake all
catalog segments into a single dense matrix at startup so query time is one
matmul + one max-per-track scatter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlatCatalog:
    """Catalog representation optimized for the `/neighbors` hot path.

    Built once at startup by `build_flat_catalog`. Holds:
      - `track_ids`: ordered list of track_id strings (length N).
      - `means`: (N, CLAP_EMBED_DIM) float32 L2-normalized mean-pooled vectors.
      - `segs_flat`: (sum_of_windows, CLAP_EMBED_DIM) float32 L2-normalized
        per-window vectors, concatenated track-by-track in the order of
        `track_ids`.
      - `seg_ranges`: list of (start, end) tuples (length N) indexing into
        `segs_flat` for each track. `segs_flat[start:end]` are that track's
        per-window vectors.
    """

    track_ids: list[str]
    means: np.ndarray              # shape (N, embed_dim), float32, L2-normalized rows
    segs_flat: np.ndarray          # shape (total_segments, embed_dim), float32
    seg_ranges: list[tuple[int, int]]  # length N


def build_flat_catalog(
    catalog_tracks: list[dict],
    embeddings: np.ndarray,
    segment_embeddings: dict[str, np.ndarray],
) -> FlatCatalog:
    """Build the FlatCatalog from the loaded corpus artifacts.

    Args:
        catalog_tracks: list of dicts from corpus.json. Each row has a
            `track_id` field; row order MUST match the row order of `embeddings`.
        embeddings: array loaded from embeddings.npy, shape (N, embed_dim).
        segment_embeddings: dict from segment_embeddings.npz, keyed by track_id.
            Each value is shape (num_windows_for_that_track, embed_dim).

    Returns:
        A FlatCatalog with `track_ids`, `means`, `segs_flat`, `seg_ranges`
        populated and aligned. Row 0 of `means` and `segs_flat[seg_ranges[0]]`
        both belong to `track_ids[0]`.

    Raises:
        ValueError if any of:
          - `len(catalog_tracks) != embeddings.shape[0]`
          - a track_id present in catalog_tracks is missing from segment_embeddings
          - segment row count for a track is 0

    The startup lifespan calls this once. After that the FlatCatalog is a
    module-level constant for the request hot path.
    """
    means = np.asarray(embeddings, dtype=np.float32)
    if means.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {means.shape}")
    if len(catalog_tracks) != means.shape[0]:
        raise ValueError(
            f"catalog length {len(catalog_tracks)} does not match embeddings rows {means.shape[0]}"
        )

    track_ids: list[str] = []
    seg_arrays: list[np.ndarray] = []
    seg_ranges: list[tuple[int, int]] = []
    cursor = 0
    for row in catalog_tracks:
        track_id = row.get("track_id")
        if not track_id:
            raise ValueError("catalog row missing track_id")
        track_id = str(track_id)
        if track_id not in segment_embeddings:
            raise ValueError(f"missing segment embeddings for {track_id}")

        segs = np.asarray(segment_embeddings[track_id], dtype=np.float32)
        if segs.ndim != 2:
            raise ValueError(f"{track_id}: segment embeddings must be 2-D, got {segs.shape}")
        if segs.shape[0] == 0:
            raise ValueError(f"{track_id}: segment embeddings must have at least one row")
        if segs.shape[1] != means.shape[1]:
            raise ValueError(
                f"{track_id}: segment dim {segs.shape[1]} does not match embedding dim {means.shape[1]}"
            )

        start = cursor
        cursor += segs.shape[0]
        seg_ranges.append((start, cursor))
        seg_arrays.append(segs)
        track_ids.append(track_id)

    segs_flat = np.vstack(seg_arrays).astype(np.float32) if seg_arrays else np.empty((0, means.shape[1]), dtype=np.float32)
    return FlatCatalog(track_ids=track_ids, means=means, segs_flat=segs_flat, seg_ranges=seg_ranges)


def top_k_neighbors(
    query_mean: np.ndarray,
    query_segs: np.ndarray,
    catalog: FlatCatalog,
    k: int = 5,
) -> list[dict]:
    """Rank catalog tracks against the query; return top-k with both similarity metrics.

    Args:
        query_mean: shape (embed_dim,), float32, L2-normalized. Track-level
            mean-pooled embedding of the query audio.
        query_segs: shape (Q, embed_dim), float32, rows L2-normalized. Q ≥ 1.
            Per-window embeddings of the query.
        catalog: prebuilt FlatCatalog.
        k: how many neighbors to return. Clamped to len(catalog.track_ids).

    Returns:
        List of length min(k, N) sorted by `meanPooledSimilarity` descending.
        Each entry: {
            "trackId": str,
            "meanPooledSimilarity": float,   # cosine, [-1, 1]; typically [0, 1] in CLAP music space
            "maxSegmentSimilarity": float,   # cosine over all (i, j) segment pairs
        }

    Notes:
        - Both metrics are returned as raw cosines, not percentages. The
          frontend converts to "87%" by rounding(sim * 100).
        - Ranking is on `meanPooledSimilarity` only. `maxSegmentSimilarity` is
          secondary and only displayed, never used to reorder.
    """
    n = len(catalog.track_ids)
    if n == 0:
        return []

    query_mean_arr = np.asarray(query_mean, dtype=np.float32)
    query_segs_arr = np.asarray(query_segs, dtype=np.float32)
    if query_mean_arr.ndim != 1:
        raise ValueError(f"query_mean must be 1-D, got shape {query_mean_arr.shape}")
    if query_segs_arr.ndim != 2 or query_segs_arr.shape[0] == 0:
        raise ValueError(f"query_segs must be non-empty 2-D, got shape {query_segs_arr.shape}")
    if query_mean_arr.shape[0] != catalog.means.shape[1]:
        raise ValueError(
            f"query dim {query_mean_arr.shape[0]} does not match catalog dim {catalog.means.shape[1]}"
        )
    if query_segs_arr.shape[1] != catalog.segs_flat.shape[1]:
        raise ValueError(
            f"query segment dim {query_segs_arr.shape[1]} does not match catalog dim {catalog.segs_flat.shape[1]}"
        )

    mean_sims = catalog.means @ query_mean_arr
    seg_sims_full = query_segs_arr @ catalog.segs_flat.T
    max_seg_sims = np.empty(n, dtype=np.float32)
    # Track which (query_window, catalog_window) pair produced the maxSegmentSimilarity
    # so the UI can show "the part of the query that matched is 0:30-0:40,
    # the part of the catalog track it matched to is 0:50-1:00."
    match_q_win = np.empty(n, dtype=np.int32)
    match_c_win = np.empty(n, dtype=np.int32)
    for i, (start, end) in enumerate(catalog.seg_ranges):
        sub = seg_sims_full[:, start:end]
        flat_idx = int(sub.argmax())
        qi, cj = np.unravel_index(flat_idx, sub.shape)
        max_seg_sims[i] = float(sub[qi, cj])
        match_q_win[i] = int(qi)
        match_c_win[i] = int(cj)

    k = max(1, min(int(k), n))
    if k >= n:
        idx = np.argsort(-mean_sims)
    else:
        idx = np.argpartition(mean_sims, -k)[-k:]
        idx = idx[np.argsort(-mean_sims[idx])]

    # WINDOW_SECONDS is the same 10 s contract used at ingest + query time.
    # Importing the constant here is overkill since the UI mostly needs the
    # window index; the frontend multiplies by 10 to render MM:SS.
    return [
        {
            "trackId": catalog.track_ids[int(i)],
            "meanPooledSimilarity": float(mean_sims[int(i)]),
            "maxSegmentSimilarity": float(max_seg_sims[int(i)]),
            "matchQueryWindow": int(match_q_win[int(i)]),
            "matchCatalogWindow": int(match_c_win[int(i)]),
        }
        for i in idx
    ]


# Above this catalog size, the full N×N pairwise computation is replaced by a
# deterministic K-row sample. The exact path materializes an N×N cosine matrix
# plus np.triu_indices (two int64 arrays of N*(N-1)/2) plus the off-diagonal
# vector — ~33 GB at N≈49k, which OOM-crashes the Space at startup. A sample of
# K rows yields a statistically representative pairwise-cosine distribution for
# percentile calibration at O(K²) memory (~150 MB at K=6000).
CATALOG_DISTRIBUTION_MAX_SAMPLE = 6000


def compute_catalog_distribution(catalog: FlatCatalog) -> np.ndarray:
    """Sort the pairwise catalog-cosine distribution (excluding self-pairs).

    Used to calibrate the user-facing similarity score per ADR-0001. CLAP music
    embeddings cluster tightly (anisotropy), so raw cosine doesn't map cleanly
    to a percentage — instead we map each query-vs-track cosine to its percentile
    rank in this distribution. Computed once at startup.

    For catalogs larger than ``CATALOG_DISTRIBUTION_MAX_SAMPLE`` the distribution
    is computed from a deterministic random sample of rows (seeded, reproducible)
    to bound memory; the exact all-pairs path is kept for smaller catalogs.

    Returns:
        1-D float32 array of off-diagonal upper-triangle pairwise cosines, sorted
        ascending — length N*(N-1)/2 for small N, else K*(K-1)/2. Empty if N < 2.
    """
    n = len(catalog.track_ids)
    if n < 2:
        return np.empty((0,), dtype=np.float32)
    means = catalog.means
    if n > CATALOG_DISTRIBUTION_MAX_SAMPLE:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=CATALOG_DISTRIBUTION_MAX_SAMPLE, replace=False)
        means = means[idx]
        n = CATALOG_DISTRIBUTION_MAX_SAMPLE
    sim = means @ means.T
    iu = np.triu_indices(n, k=1)
    off_diag = sim[iu].astype(np.float32)
    off_diag.sort()
    return off_diag


def cosine_to_percentile(cosine: float, sorted_distribution: np.ndarray) -> float:
    """Map a raw cosine to a percentile rank in the catalog distribution.

    Returns:
        Float in [0.0, 1.0] — fraction of catalog-vs-catalog pairs that score
        BELOW the given cosine. 1.0 means this match is more similar than every
        observed catalog-pair similarity; 0.0 means it's below the floor.

    Edge cases:
        - Empty distribution → returns 0.5 (no information; render as moderate).
        - Cosine above max in distribution → returns 1.0.
        - Cosine below min in distribution → returns 0.0.
    """
    if sorted_distribution.size == 0:
        return 0.5
    idx = int(np.searchsorted(sorted_distribution, float(cosine), side="left"))
    return idx / float(sorted_distribution.size)


def similarity_label(percentile_rank: float) -> str:
    """Return a coarse human-readable label for a percentile rank.

    Thresholds per ADR-0001. Reviewable as the catalog grows.
    """
    p = float(percentile_rank)
    if p >= 0.95:
        return "very close"
    if p >= 0.80:
        return "close"
    if p >= 0.50:
        return "moderate"
    return "weak"


def query_specificity(query_mean: np.ndarray, catalog: FlatCatalog, threshold: float = 0.95) -> float:
    """Score how specific (vs generic) a query is against the catalog.

    A query that scores above `threshold` against most of the catalog is broadly
    similar to many tracks — generic. A query that exceeds the threshold against
    only a handful is specific.

    Returns:
        Float in [0.0, 1.0]. 0.0 = maximally generic (matches everything);
        1.0 = maximally specific (matches almost nothing above threshold).

    Used in the UI to render a "this generation pattern is broadly similar to
    many tracks" note when the query is generic.
    """
    n = len(catalog.track_ids)
    if n == 0:
        return 1.0
    sims = catalog.means @ np.asarray(query_mean, dtype=np.float32)
    above = int((sims >= float(threshold)).sum())
    return 1.0 - (above / float(n))


# --- ADR-0004: per-criterion comparison helpers ------------------------------
#
# Each helper takes one query value + one catalog value (the shape matches
# what MirFeatures stores) and returns a dict with `agreement` (0-1 float for
# UI bar widths) and `label` (categorical for the user-facing readout).
#
# Per-criterion thresholds are locked in ADR-0004 §"The per-criterion
# thresholds (locked here, not adjustable per deploy)". Changing them
# requires an ADR amendment, not a quiet config tweak.


def compare_tempos(query_bpm: float, match_bpm: float) -> dict:
    """Pairwise tempo agreement. Symmetric, agreement decays with |Δ BPM|."""
    q = float(query_bpm or 0.0)
    m = float(match_bpm or 0.0)
    delta = abs(q - m)
    if delta <= 3.0:
        return {"agreement": 1.0, "label": "same tempo"}
    if delta <= 10.0:
        return {"agreement": float(max(0.0, 1.0 - delta / 20.0)), "label": f"{round(delta)} BPM apart"}
    return {"agreement": float(max(0.0, 1.0 - delta / 40.0)), "label": f"{round(delta)} BPM apart"}


# Pitch-class index for fifth + relative key relationships.
_PITCH_INDEX = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def compare_keys(query_key: str, query_mode: str, match_key: str, match_mode: str) -> dict:
    """Pairwise key + mode agreement using the music-theoretic relationships
    that are most legible to a non-technical user: same key, relative major/
    minor, perfect-fifth-apart, otherwise different.
    """
    q_idx = _PITCH_INDEX.get(str(query_key), 0)
    m_idx = _PITCH_INDEX.get(str(match_key), 0)
    q_mode = str(query_mode).lower()
    m_mode = str(match_mode).lower()

    if q_idx == m_idx and q_mode == m_mode:
        return {"agreement": 1.0, "label": "same key"}

    # Relative major <-> minor: e.g. C major <-> A minor share all 7 pitches.
    # A minor is 3 semitones below C major's tonic; that's a 9-up or 3-down
    # semitone relationship between the two tonics, mode swap.
    if q_mode != m_mode:
        diff = (m_idx - q_idx) % 12
        if (q_mode == "major" and m_mode == "minor" and diff == 9) or \
           (q_mode == "minor" and m_mode == "major" and diff == 3):
            return {"agreement": 0.7, "label": "relative key"}

    # Perfect-fifth relationship: tonics 7 semitones apart (or 5 down), same mode.
    if q_mode == m_mode:
        diff = (m_idx - q_idx) % 12
        if diff in (5, 7):
            return {"agreement": 0.5, "label": "fifth apart"}

    return {"agreement": 0.0, "label": "different key"}


def compare_chroma_vectors(query: list | np.ndarray, match: list | np.ndarray) -> dict:
    """Pairwise cosine on the 12-d chroma mean vector → categorical label.

    Maps 'how much do these tracks share their chord/pitch palette?' to a
    user-facing string. Cosine on probability-normalized chroma is bounded
    by construction; we surface it as agreement directly.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    m = np.asarray(match, dtype=np.float32).reshape(-1)
    if q.size == 0 or m.size == 0:
        return {"agreement": 0.0, "label": "different chord palette"}
    qn = float(np.linalg.norm(q)) or 1.0
    mn = float(np.linalg.norm(m)) or 1.0
    cos = float(np.dot(q, m) / (qn * mn))
    cos = max(0.0, min(1.0, cos))  # clamp negatives + numerical drift
    if cos >= 0.85:
        return {"agreement": cos, "label": "very similar chord palette"}
    if cos >= 0.65:
        return {"agreement": cos, "label": "similar chord palette"}
    if cos >= 0.40:
        return {"agreement": cos, "label": "moderate chord overlap"}
    return {"agreement": cos, "label": "different chord palette"}


def compare_timbre_vectors(query: list | np.ndarray, match: list | np.ndarray) -> dict:
    """Pairwise cosine on the 26-d MFCC mean+std vector → 'production feel' label.

    MFCC captures the spectral envelope (instruments, mix, mastering) more
    than melody or harmony. Two tracks with similar MFCC fingerprints share
    a sonic aesthetic even if pitched differently.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    m = np.asarray(match, dtype=np.float32).reshape(-1)
    if q.size == 0 or m.size == 0:
        return {"agreement": 0.0, "label": "different production"}
    qn = float(np.linalg.norm(q)) or 1.0
    mn = float(np.linalg.norm(m)) or 1.0
    cos = float(np.dot(q, m) / (qn * mn))
    # MFCC vectors can be anti-correlated → cosine in [-1, 1]. Clamp to [0, 1]
    # for the agreement bar — negative cosine reads as "different" not "less."
    agreement = max(0.0, min(1.0, cos))
    if agreement >= 0.80:
        return {"agreement": agreement, "label": "very similar production feel"}
    if agreement >= 0.55:
        return {"agreement": agreement, "label": "similar production feel"}
    if agreement >= 0.25:
        return {"agreement": agreement, "label": "moderately different production"}
    return {"agreement": agreement, "label": "different production"}


def threshold_from_manifest(manifest: dict) -> float:
    """Read the `threshold_default` field from the parsed manifest.json.

    Returns:
        The float threshold (LOCKED_DECISIONS provisional default: 0.70).

    Raises:
        KeyError if the manifest lacks `threshold_default` — this is a hard
        invariant of the Phase 1 manifest schema and a missing field indicates
        a stale/broken corpus build.

    Used by /neighbors to include `thresholdDefault` in the response so the
    frontend and backend agree on the cutoff for the "Completely unique"
    headline rule.
    """
    return float(manifest["threshold_default"])
