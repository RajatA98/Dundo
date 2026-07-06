"""Spike: tag catalog tracks with genre/mood/instrument from MuQ embeddings — NO pod, NO audio."""
import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from backend import muq_engine

REV = "3f82ace98dfa0d18c1ac025eb6202ec4beeeb80d"
emb_path = hf_hub_download("RajatA98/dundo-corpus", "embeddings.npy", repo_type="dataset", revision=REV)
corpus_path = hf_hub_download("RajatA98/dundo-corpus", "corpus.json", repo_type="dataset", revision=REV)
embeddings = np.load(emb_path).astype(np.float32)
tracks = json.load(open(corpus_path))
print(f"CATALOG embeddings={embeddings.shape} tracks={len(tracks)}", flush=True)

print("loading MuQ-MuLan (first run downloads ~2.8GB)...", flush=True)
muq_engine.load()
print("MuQ loaded", flush=True)


def encode_texts(prompts):
    with torch.no_grad():
        te = muq_engine._model(texts=prompts).detach().cpu().numpy().astype(np.float32)
    te = te / np.maximum(np.linalg.norm(te, axis=1, keepdims=True), 1e-12)
    return te


MOODS = ["upbeat", "mellow", "atmospheric", "dark", "energetic", "dreamy", "melancholic", "driving"]
INSTR = ["synth-heavy", "guitar-led", "piano-led", "vocal-led", "drum-heavy", "acoustic", "orchestral", "electronic"]
mood_emb = encode_texts([f"a {m} music track" for m in MOODS])
instr_emb = encode_texts([f"a {i} music track" for i in INSTR])

idxs = np.linspace(0, len(tracks) - 1, 18).astype(int)
print("\n--- SAMPLE TAGS (genre / mood / instrument, all from MuQ embedding) ---", flush=True)
for i in idxs:
    e = embeddings[i]
    g = [x[0] for x in muq_engine.top_genres(e, k=2)]
    mtop = [MOODS[j] for j in np.argsort(-(mood_emb @ e))[:2]]
    itop = [INSTR[j] for j in np.argsort(-(instr_emb @ e))[:2]]
    t = tracks[i]
    artist = (t.get("artist") or "?")[:24]
    title = (t.get("title") or "?")[:24]
    print(f"{artist:24} | {title:24} | {g} · {mtop} · {itop}", flush=True)
print("SPIKE_DONE", flush=True)
