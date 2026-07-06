"""Spike v3: derive upload tags from acoustic k-NN's REAL MTG tags (tag propagation), vs zero-shot.

For each held-out catalog track, predict its tags from the similarity-weighted vote of its top-N
nearest neighbors' editorial tags (excluding self + same artist). Measure recovery vs its own truth.
This validates the upload-side WITHOUT zero-shot guessing.
"""
import json
import numpy as np
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

REV = "3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d"
K = 3
NEIGHBORS = 25

SUPER = {
    "rock": "rock", "hardrock": "rock", "poprock": "rock", "postrock": "rock", "punkrock": "rock",
    "instrumentalrock": "rock", "indie": "rock", "alternative": "rock", "progressive": "rock", "psychedelic": "rock",
    "electronic": "electronic", "edm": "electronic", "house": "electronic", "deephouse": "electronic",
    "techno": "electronic", "trance": "electronic", "dubstep": "electronic", "drumnbass": "electronic",
    "breakbeat": "electronic", "idm": "electronic", "minimal": "electronic", "club": "electronic",
    "dance": "electronic", "electropop": "electronic", "synthpop": "electronic", "eurodance": "electronic",
    "dub": "electronic", "triphop": "electronic",
    "pop": "pop", "instrumentalpop": "pop", "popfolk": "pop",
    "classical": "classical", "orchestral": "classical", "symphonic": "classical", "contemporary": "classical",
    "medieval": "classical", "newage": "classical",
    "jazz": "jazz", "jazzfusion": "jazz", "fusion": "jazz", "swing": "jazz", "bossanova": "jazz", "lounge": "jazz",
    "folk": "folk", "celtic": "folk", "country": "folk", "ethno": "folk", "world": "folk", "tribal": "folk", "latin": "folk",
    "hiphop": "hiphop", "rap": "hiphop", "rnb": "hiphop", "soul": "hiphop", "funk": "hiphop", "groove": "hiphop",
    "ambient": "ambient", "atmospheric": "ambient", "chillout": "ambient", "experimental": "ambient", "improvisation": "ambient",
    "metal": "metal", "hard": "metal", "blues": "blues", "reggae": "reggae", "disco": "disco", "soundtrack": "soundtrack",
}

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
            tags = [x.split("---")[-1] for x in c[5:] if "---" in x]
            if tags:
                out[tid] = tags
    return out

genre = parse("/tmp/mtg_autotagging_genre.tsv")
instr = parse("/tmp/mtg_autotagging_instrument.tsv")
emb = np.load(hf_hub_download("RajatA98/dundo-corpus", "embeddings.npy", repo_type="dataset", revision=REV)).astype(np.float32)
tracks = json.load(open(hf_hub_download("RajatA98/dundo-corpus", "corpus.json", repo_type="dataset", revision=REV)))
jid = np.array([(int(t["external_ids"]["jamendoTrackId"]) if t.get("external_ids", {}).get("jamendoTrackId", "").isdigit() else -1) for t in tracks])
artist = [t.get("artist") for t in tracks]
emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

def knn_eval(name, gtmap, coarse):
    have = np.array([i for i, j in enumerate(jid) if j in gtmap])
    def truth(i):
        s = set(gtmap[jid[i]])
        return {SUPER.get(x, x) for x in s} if coarse else s
    rng = np.random.default_rng(0)
    sample = rng.choice(have, size=min(2500, len(have)), replace=False)
    p, cov, n = [], 0, 0
    for i in sample:
        sims = emb @ emb[i]
        sims[i] = -2
        order = np.argpartition(-sims, NEIGHBORS + 40)[:NEIGHBORS + 40]
        order = order[np.argsort(-sims[order])]
        votes = defaultdict(float); used = 0
        for nb in order:
            if jid[nb] not in gtmap or artist[nb] == artist[i]:
                continue
            for tg in truth(nb) if False else (gtmap[jid[nb]]):
                lab = SUPER.get(tg, tg) if coarse else tg
                votes[lab] += float(sims[nb])
            used += 1
            if used >= NEIGHBORS:
                break
        preds = {lab for lab, _ in sorted(votes.items(), key=lambda x: -x[1])[:K]}
        tset = truth(i)
        if not tset:
            continue
        n += 1
        h = len(preds & tset)
        p.append(h / K)
        cov += 1 if h else 0
    print(f"[kNN {name}] precision@{K}={np.mean(p):.3f} coverage@{K}={cov/n:.3f} (n={n}, neighbors={NEIGHBORS})", flush=True)

knn_eval("COARSE GENRE", genre, True)
knn_eval("FINE GENRE", genre, False)
knn_eval("INSTRUMENT", instr, False)
print("KNN_DONE", flush=True)
