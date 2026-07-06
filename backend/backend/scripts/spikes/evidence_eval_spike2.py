"""Spike v2: recovery at the granularity we'd actually SHOW — coarse super-genres + mood + instrument."""
import json
import numpy as np
import torch
from collections import Counter
from huggingface_hub import hf_hub_download
from backend import muq_engine

REV = "3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d"
K = 3

SUPER = {
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
    "folk": "folk", "celtic": "folk", "country": "folk", "ethno": "folk", "world": "folk",
    "tribal": "folk", "latin": "folk",
    "hiphop": "hiphop", "rap": "hiphop", "rnb": "hiphop", "soul": "hiphop", "funk": "hiphop", "groove": "hiphop",
    "ambient": "ambient", "atmospheric": "ambient", "chillout": "ambient", "experimental": "ambient",
    "improvisation": "ambient", "meditative": "ambient", "soundscape": "ambient",
    "metal": "metal", "hard": "metal",
    "blues": "blues", "reggae": "reggae", "disco": "disco", "soundtrack": "soundtrack", "film": "soundtrack",
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
mood = parse("/tmp/mtg_autotagging_moodtheme.tsv")
instr = parse("/tmp/mtg_autotagging_instrument.tsv")
emb = np.load(hf_hub_download("RajatA98/dundo-corpus", "embeddings.npy", repo_type="dataset", revision=REV)).astype(np.float32)
tracks = json.load(open(hf_hub_download("RajatA98/dundo-corpus", "corpus.json", repo_type="dataset", revision=REV)))
jid = [(int(t["external_ids"]["jamendoTrackId"]) if t.get("external_ids", {}).get("jamendoTrackId", "").isdigit() else -1) for t in tracks]

muq_engine.load()
def enc(prompts):
    with torch.no_grad():
        te = muq_engine._model(texts=prompts).detach().cpu().numpy().astype(np.float32)
    return te / np.maximum(np.linalg.norm(te, axis=1, keepdims=True), 1e-12)

def run(name, gtmap, prompt, truth_xform=lambda s: set(s), min_support=50):
    rows = [i for i, j in enumerate(jid) if j in gtmap]
    truth = [truth_xform(gtmap[jid[i]]) for i in rows]
    cnt = Counter(g for s in truth for g in s)
    vocab = sorted([g for g, n in cnt.items() if n >= min_support])
    E = emb[rows]
    T = enc([prompt.format(g) for g in vocab])
    S = E @ T.T
    topk = np.argsort(-S, axis=1)[:, :K]
    p, cov, n = [], 0, 0
    for r in range(len(rows)):
        preds = {vocab[j] for j in topk[r]}
        tset = truth[r] & set(vocab)
        if not tset:
            continue
        n += 1
        h = len(preds & tset)
        p.append(h / K)
        cov += 1 if h else 0
    print(f"[{name}] vocab={len(vocab)} precision@{K}={np.mean(p):.3f} coverage@{K}={cov/n:.3f} (n={n})", flush=True)

run("COARSE GENRE", genre, "a {} music track", truth_xform=lambda s: {SUPER.get(x, x) for x in s})
run("MOOD/THEME", mood, "a {} song", min_support=50)
run("INSTRUMENT", instr, "a song with {}", min_support=50)
print("EVAL2_DONE", flush=True)
