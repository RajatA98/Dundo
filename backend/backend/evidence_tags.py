"""Evidence Layer — assemble the per-match `evidenceTags` overlap block.

Validated design (see factory/artifacts/CODEX_EVIDENCE_LAYER_BUILD_SIGNOFF.md + the
`scripts/_evidence_*` spikes):
  - Artist/match descriptors = REAL MTG-Jamendo editorial tags (authoritative).
  - Upload descriptors = similarity-weighted vote of the upload's acoustic neighbors' REAL tags
    (k-NN tag propagation), EXCLUDING the displayed candidate's artist (no circular evidence).
  - shared = gated(query vote-share >= tau) ∩ candidate's real tags.
  - tau=0.30 from the held-out sweep (~0.77 display precision / 0.90 coverage on coarse genre).
  - No softmax (multi-label); calibration is the weighted vote; descriptors below tau are hidden.

Pure + dependency-light so it is unit-testable without the model or catalog load.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

METHOD = "mtg-knn-v1"
TAU_DEFAULT = 0.30
POOL_DEFAULT = 25
SUPPORT_MIN = 50  # labels below this catalog support are excluded upstream (in catalog_tags asset)

# Which catalog-tag kinds feed the overlap, and the field they read from catalog_tags entries.
TAG_KINDS = (("genre", "coarseGenre"), ("instrument", "instrument"), ("mood", "mood"))


@dataclass(frozen=True)
class Neighbor:
    """One acoustic neighbor of the upload (already excludes the upload itself)."""
    track_id: str
    artist: str | None
    sim: float


def _flatten(entry: dict | None) -> list[tuple[str, str]]:
    if not entry:
        return []
    out: list[tuple[str, str]] = []
    for kind, field in TAG_KINDS:
        for label in entry.get(field, []) or []:
            out.append((kind, label))
    return out


def _confidence(share: float, tau: float) -> str:
    if share >= max(0.40, tau + 0.10):
        return "high"
    if share >= tau:
        return "medium"
    return "low"


def assemble_evidence_tags(
    neighbors: list[Neighbor],
    candidate_track_id: str,
    candidate_artist: str | None,
    catalog_tags: dict[str, dict],
    *,
    tau: float = TAU_DEFAULT,
    pool: int = POOL_DEFAULT,
) -> dict | None:
    """Build the evidenceTags block for one match, or None when nothing clears the gate.

    Args:
        neighbors: upload's acoustic neighbors, ranked by sim desc (NOT including the upload).
        candidate_track_id / candidate_artist: the match we are explaining the overlap with.
        catalog_tags: {track_id: {"coarseGenre":[...], "genre":[...], "instrument":[...], "mood":[...]}}.
        tau: minimum query vote-share for a descriptor to count (default 0.30 from the sweep).
        pool: number of (post-exclusion) tagged neighbors to vote over.

    Returns:
        A dict matching the `evidenceTags` contract (shared/query/match/confidence/method +
        provenance), or None to hide the block (no trustworthy overlap).
    """
    match_tags = _flatten(catalog_tags.get(candidate_track_id))
    match_set = set(match_tags)

    votes: dict[tuple[str, str], float] = defaultdict(float)
    total = 0.0
    used = 0
    for nb in neighbors:
        if candidate_artist is not None and nb.artist == candidate_artist:
            continue  # circularity guard: exclude the candidate's own artist from the pool
        entry = catalog_tags.get(nb.track_id)
        if not entry:
            continue
        w = float(nb.sim)
        if w <= 0:
            continue
        total += w
        for kl in _flatten(entry):
            votes[kl] += w
        used += 1
        if used >= pool:
            break

    if total <= 0 or not votes:
        return None

    shares = {kl: v / total for kl, v in votes.items()}
    query = [
        {"kind": k, "label": l, "confidence": round(s, 3)}
        for (k, l), s in sorted(shares.items(), key=lambda x: -x[1])
        if s >= tau
    ]
    shared = [t for t in query if (t["kind"], t["label"]) in match_set]
    if not shared:
        return None  # honest fallback — hide the block rather than assert a weak overlap

    block_conf = "high" if len(shared) >= 2 else ("medium" if shared[0]["confidence"] >= 0.40 else "low")
    return {
        "shared": shared,
        "query": query,
        "match": [{"kind": k, "label": l} for k, l in match_tags],
        "confidence": block_conf,
        "method": METHOD,
        "neighborCount": used,
        "excludedCandidate": candidate_artist is not None,
    }
