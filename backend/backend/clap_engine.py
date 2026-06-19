"""CLAP-based genre tagging + audio embeddings.

A singleton module that loads `laion/larger_clap_music` once and exposes:
  - `load()` — idempotent; called from the CLI and the FastAPI lifespan.
  - `encode_audio(wav_mono, sr) -> np.ndarray(1024,)` — L2-normalized embedding.
  - `top_genres(emb, k=3) -> [(label, prob)]` — zero-shot via cached text embeddings.

The same embedding feeds Phase 2's similarity search; Phase 1 consumes only the
genre tag. One model load, two jobs.

Torch + transformers imports are deferred to inside `load()` so the rest of the
package (signals, scoring, librosa_engine, corruptions, cli score command for
non-genre paths) doesn't pay the import cost.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from . import config

_model: Any = None
_processor: Any = None
_genre_text_emb: np.ndarray | None = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()


def load() -> None:
    """Load CLAP and cache the genre text embeddings. Idempotent and thread-safe."""
    global _model, _processor, _genre_text_emb
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        # Deferred imports — keep tooling startup snappy when CLAP isn't needed.
        import torch
        from transformers import ClapModel, ClapProcessor

        proc = ClapProcessor.from_pretrained(config.CLAP_MODEL_ID)
        mdl = ClapModel.from_pretrained(config.CLAP_MODEL_ID).eval()

        # Pre-encode the genre prompts once at startup, L2-normalized so the
        # similarity at call time is a pure dot product. We use the explicit
        # text_model → text_projection path instead of `get_text_features`,
        # which in transformers 5.9.x returns a wrapped output object rather
        # than a tensor — the explicit path is robust across versions.
        prompts = [config.GENRE_PROMPT_TEMPLATE.format(label=lbl) for lbl in config.GENRE_LABELS]
        text_inputs = proc(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_out = mdl.text_model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
            text_feats = (
                mdl.text_projection(text_out.pooler_output)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        text_feats /= np.maximum(
            np.linalg.norm(text_feats, axis=1, keepdims=True), 1e-12
        )

        _processor = proc
        _model = mdl
        _genre_text_emb = text_feats


def encode_audio(wav_mono: np.ndarray, sr: int) -> np.ndarray:
    """Return the L2-normalized 1024-d audio embedding for a mono signal."""
    if _model is None:
        load()

    # CLAP expects 48 kHz mono.
    if sr != config.CLAP_SR:
        import librosa
        wav_mono = librosa.resample(
            wav_mono.astype(np.float32), orig_sr=sr, target_sr=config.CLAP_SR
        )
        sr = config.CLAP_SR

    import torch

    with _encode_lock:  # CPU torch isn't reliably thread-safe.
        # `audio=` not `audios=` (deprecated in transformers ≥5.x).
        inputs = _processor(audio=wav_mono, sampling_rate=sr, return_tensors="pt")
        # Explicit projection path; `no_grad` scoped here so the test_client
        # lifespan exit can't leak a torch grad state back into the encoder.
        with torch.no_grad():
            audio_out = _model.audio_model(input_features=inputs.input_features)
            feats = (
                _model.audio_projection(audio_out.pooler_output)
                .cpu()
                .numpy()
                .astype(np.float32)[0]
            )

    feats /= max(float(np.linalg.norm(feats)), 1e-12)
    return feats


def top_genres(
    emb: np.ndarray, k: int = config.CLAP_GENRE_TOP_K
) -> list[tuple[str, float]]:
    """Zero-shot genre tagging: cosine sim → temperature-scaled softmax → top-k."""
    if _genre_text_emb is None:
        load()
    sims = _genre_text_emb @ emb            # both L2-normalized → cosine similarity
    scaled = sims * config.CLAP_GENRE_TEMPERATURE
    exp = np.exp(scaled - float(scaled.max()))
    probs = exp / exp.sum()
    idx = np.argsort(probs)[::-1][:k]
    return [(config.GENRE_LABELS[int(i)], float(probs[int(i)])) for i in idx]


def is_loaded() -> bool:
    return _model is not None


def model_id() -> str:
    return config.CLAP_MODEL_ID
