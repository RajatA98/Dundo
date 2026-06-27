from backend.evidence_tags import Neighbor, assemble_evidence_tags

CATALOG = {
    "cand": {"coarseGenre": ["rock"], "instrument": ["guitar"]},
    "n1": {"coarseGenre": ["rock"], "instrument": ["guitar"]},
    "n2": {"coarseGenre": ["rock"]},
    "n3": {"coarseGenre": ["electronic"]},
    "same": {"coarseGenre": ["jazz"]},  # belongs to candidate's artist -> must be excluded
}


def test_overlap_with_candidate_excluded():
    neighbors = [
        Neighbor("n1", "artistA", 0.9),
        Neighbor("n2", "artistB", 0.8),
        Neighbor("n3", "artistC", 0.2),
        Neighbor("same", "candArtist", 0.95),  # circularity guard target
    ]
    block = assemble_evidence_tags(neighbors, "cand", "candArtist", CATALOG, tau=0.3, pool=10)
    shared = {(t["kind"], t["label"]) for t in block["shared"]}
    assert ("genre", "rock") in shared          # strong, real on both sides
    assert ("instrument", "guitar") in shared
    assert block["excludedCandidate"] is True
    assert block["method"] == "mtg-knn-v1"
    assert block["neighborCount"] == 3          # 'same' excluded
    # the candidate-artist neighbor's tag never leaks into the query/overlap (no circular evidence)
    assert all(t["label"] != "jazz" for t in block["query"])


def test_tau_gates_weak_descriptors():
    neighbors = [Neighbor("n1", "a", 0.9), Neighbor("n3", "b", 0.85)]  # rock 0.51 vs electronic 0.49
    block = assemble_evidence_tags(neighbors, "cand", "candArtist", CATALOG, tau=0.6, pool=10)
    # neither label clears tau=0.6 -> no trustworthy overlap -> hide
    assert block is None


def test_fallback_hides_when_no_overlap():
    neighbors = [Neighbor("n3", "artistC", 0.9)]  # electronic only; candidate is rock/guitar
    assert assemble_evidence_tags(neighbors, "cand", "candArtist", CATALOG, tau=0.3) is None
