"""Slice 2 prototype: candidate-exclusion guard + τ sweep for the evidence overlap.

Simulates the live scenario with a held-out catalog track as the 'upload' (so we have ground truth):
  - candidate = top tagged neighbor (the artist we explain the match to)
  - query tags = similarity-weighted vote of OTHER neighbors, EXCLUDING the candidate's artist (no circularity)
  - shared = gated(query tags, vote-share >= tau) ∩ candidate's REAL tags
  - display precision = of shared descriptors shown, how many are TRUE of the query (in its own MTG tags)
Reports display precision / coverage / fallback across tau, split by coarse-genre + instrument.
"""
import json
import numpy as np
from collections import defaultdict
from huggingface_hub import hf_hub_download
from backend.scripts.build_catalog_tags import SUPER_GENRE

REV = "3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d"
POOL = 25
TAUS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

def parse(path):
    out = {}
    with open(path) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            try:
                tid = int(c[0].replace("track_", ""))
            except ValueError:
                continue
            tg = [x.split("---")[-1] for x in c[5:] if "---" in x]
            if tg:
                out[tid] = tg
    return out

genre_raw = parse("/tmp/mtg_tags_cache/autotagging_genre.tsv")
instr = parse("/tmp/mtg_tags_cache/autotagging_instrument.tsv")
genre = {k: sorted({SUPER_GENRE.get(g, g) for g in v}) for k, v in genre_raw.items()}  # coarse
emb = np.load(hf_hub_download("RajatA98/dundo-corpus", "embeddings.npy", repo_type="dataset", revision=REV)).astype(np.float32)
emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
tracks = json.load(open(hf_hub_download("RajatA98/dundo-corpus", "corpus.json", repo_type="dataset", revision=REV)))
jid = np.array([(int(t["external_ids"]["jamendoTrackId"]) if t.get("external_ids", {}).get("jamendoTrackId", "").isdigit() else -1) for t in tracks])
artist = [t.get("artist") for t in tracks]


def sweep(name, gt):
    have = np.array([i for i, j in enumerate(jid) if j in gt])
    rng = np.random.default_rng(0)
    sample = rng.choice(have, size=2000, replace=False)
    stat = {t: {"shown": 0, "correct": 0, "pairs": 0, "covered": 0} for t in TAUS}
    for q in sample:
        sims = emb @ emb[q]
        sims[q] = -2
        order = np.argpartition(-sims, POOL + 80)[:POOL + 80]
        order = order[np.argsort(-sims[order])]
        # candidate = top neighbor with gt tags
        cand = next((nb for nb in order if jid[nb] in gt), None)
        if cand is None:
            continue
        cand_artist = artist[cand]
        cand_tags = set(gt[jid[cand]])
        true_q = set(gt.get(jid[q], []))
        # propagation pool: neighbors with gt, EXCLUDING candidate's artist
        pool, votes, tot = [], defaultdict(float), 0.0
        for nb in order:
            if jid[nb] not in gt or artist[nb] == cand_artist:
                continue
            w = float(sims[nb])
            tot += w
            for tg in gt[jid[nb]]:
                votes[tg] += w
            pool.append(nb)
            if len(pool) >= POOL:
                break
        if tot <= 0:
            continue
        shares = {t: v / tot for t, v in votes.items()}
        for tau in TAUS:
            gated = {t for t, s in shares.items() if s >= tau}
            shared = gated & cand_tags
            stat[tau]["pairs"] += 1
            stat[tau]["shown"] += len(shared)
            stat[tau]["correct"] += len(shared & true_q)
            stat[tau]["covered"] += 1 if shared else 0
    print(f"\n=== {name} (pool={POOL}, candidate-artist excluded) ===")
    print(f"{'tau':>5} {'display_prec':>12} {'coverage':>9} {'fallback':>9}")
    for tau in TAUS:
        s = stat[tau]
        dp = s["correct"] / s["shown"] if s["shown"] else float("nan")
        cov = s["covered"] / s["pairs"]
        print(f"{tau:>5.2f} {dp:>12.3f} {cov:>9.3f} {1-cov:>9.3f}")


sweep("COARSE GENRE", genre)
sweep("INSTRUMENT", instr)
print("\nSWEEP_DONE")
