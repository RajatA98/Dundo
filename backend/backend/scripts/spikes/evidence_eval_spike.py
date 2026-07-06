"""Validation spike: does MuQ zero-shot recover MTG editorial GENRE tags? Raw vs per-label-percentile calibrated.

Ground truth = MTG-Jamendo genre tags. Predict from MuQ embeddings (no audio). Report display-precision metrics
(precision@k, coverage rate) per Codex — these gate whether the upload-side characterizer is trustworthy.
"""
import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from backend import muq_engine

REV = "3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d"
MIN_SUPPORT = 50          # Codex: minimum-support so rare labels aren't false-precision traps
K = 3

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
                out[tid] = set(tags)
    return out

gt = parse("/tmp/mtg_autotagging_genre.tsv")
emb = np.load(hf_hub_download("RajatA98/dundo-corpus", "embeddings.npy", repo_type="dataset", revision=REV)).astype(np.float32)
tracks = json.load(open(hf_hub_download("RajatA98/dundo-corpus", "corpus.json", repo_type="dataset", revision=REV)))

# Align rows -> jamendo id -> ground-truth genre set (only tracks WITH gt)
jids = []
for t in tracks:
    jid = (t.get("external_ids") or {}).get("jamendoTrackId")
    jids.append(int(jid) if jid and jid.isdigit() else -1)
rows = [i for i, j in enumerate(jids) if j in gt]
E = emb[rows]                                   # (M,512) normalized
truth = [gt[jids[i]] for i in rows]
print(f"eval tracks (with gt genre): {len(rows)}", flush=True)

# Vocabulary = MTG genres with >= MIN_SUPPORT in our catalog
from collections import Counter
cnt = Counter(g for s in truth for g in s)
VOCAB = sorted([g for g, n in cnt.items() if n >= MIN_SUPPORT])
print(f"genre vocab (support>={MIN_SUPPORT}): {len(VOCAB)} labels", flush=True)

print("loading MuQ + encoding genre prompts...", flush=True)
muq_engine.load()
def enc(prompts):
    with torch.no_grad():
        te = muq_engine._model(texts=prompts).detach().cpu().numpy().astype(np.float32)
    return te / np.maximum(np.linalg.norm(te, axis=1, keepdims=True), 1e-12)
T = enc([f"a {g} music track" for g in VOCAB])   # (V,512)

S = E @ T.T                                       # (M,V) raw cosine scores

def evaluate(scores, label):
    topk = np.argsort(-scores, axis=1)[:, :K]
    p_at_k = []; covered = 0
    for r in range(len(rows)):
        preds = {VOCAB[j] for j in topk[r]}
        tset = truth[r] & set(VOCAB)              # only labels in vocab are recoverable
        if not tset:
            continue
        hits = len(preds & tset)
        p_at_k.append(hits / K)
        covered += 1 if hits > 0 else 0
    n = len(p_at_k)
    print(f"[{label}] precision@{K}={np.mean(p_at_k):.3f}  coverage@{K}(>=1 hit)={covered/n:.3f}  (n={n})", flush=True)

evaluate(S, "RAW cosine")

# Per-label percentile calibration (Codex: gate each label by its OWN catalog distribution; no softmax)
ranks = S.argsort(axis=0).argsort(axis=0).astype(np.float32) / max(1, len(rows) - 1)   # per-column percentile
evaluate(ranks, "CALIBRATED per-label percentile")
print("EVAL_DONE", flush=True)
