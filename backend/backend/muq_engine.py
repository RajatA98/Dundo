"""MuQ-MuLan audio encoder — primary encoder per ADR-0002.

Replaces LAION-CLAP for similarity. Same public surface as `clap_engine.py`
so `clap_windowed.py` and `api.py` only have to swap the import:

  - `load()`                       — idempotent, thread-safe model load.
  - `encode_audio(wav_mono, sr)`   — returns (512,) float32, L2-normalized.
  - `top_genres(emb, k=3)`         — zero-shot via cached text embeddings.

MuQ-MuLan ingests audio at 24 kHz (CLAP was at 48 kHz), so we resample
internally — callers can pass any sample rate. The output is already
L2-normalized by the upstream model; we re-normalize defensively after
the dtype conversion to float32, but `np.linalg.norm` typically returns
1.0 ± 1e-7 right out of the model.

Genre tagging is implemented via the same CLIP-style joint embedding pattern
CLAP used: pre-compute text embeddings of the genre prompts at startup, then
take the dot product with each query embedding at inference. MuQ-MuLan is
text-music joint, so this path is natural.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from . import config

_model: Any = None
_genre_text_emb: np.ndarray | None = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()


def load() -> None:
    """Load MuQ-MuLan and cache the genre text embeddings. Idempotent + thread-safe."""
    global _model, _genre_text_emb
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        # Deferred imports keep tooling startup snappy when the encoder isn't needed.
        import torch
        from muq import MuQMuLan

        mdl = MuQMuLan.from_pretrained(config.AUDIO_ENCODER_MODEL_ID).eval()

        # Pre-encode genre prompts once at startup. MuQ-MuLan is CLIP-style:
        # the text encoder produces 512-d L2-normalized embeddings in the
        # same joint space as the audio encoder. Cosine vs audio_emb gives a
        # zero-shot classification score.
        try:
            prompts = [f"This is a {g} song." for g in config.GENRE_LABELS]
            with torch.no_grad():
                text_emb = mdl(texts=prompts)
            text_emb_np = text_emb.detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(text_emb_np, axis=1, keepdims=True)
            text_emb_np = text_emb_np / np.maximum(norms, 1e-12)
            _genre_text_emb = text_emb_np
        except Exception as exc:
            # If text-encoder path on this checkpoint differs, fall back to disabled genres.
            print(f"[muq_engine] genre prompt encoding failed ({exc!r}); top_genres will return [].")
            _genre_text_emb = None

        _model = mdl


def encode_audio(wav_mono: np.ndarray, sr: int) -> np.ndarray:
    """Encode mono audio → 512-d L2-normalized float32 embedding.

    Args:
        wav_mono: 1-D mono float audio at any sample rate.
        sr:       sample rate of `wav_mono`. Resampled internally to 24 kHz.

    Returns:
        np.ndarray shape (512,), dtype float32, L2-normalized.
    """
    load()
    import librosa
    import torch

    wav = np.asarray(wav_mono, dtype=np.float32).reshape(-1)
    if sr != config.AUDIO_ENCODER_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.AUDIO_ENCODER_SAMPLE_RATE)

    wavs = torch.from_numpy(wav).unsqueeze(0)  # shape (1, num_samples)
    with _encode_lock, torch.no_grad():
        emb = _model(wavs=wavs)
    arr = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n > 0:
        arr = arr / n
    return arr


def top_genres(emb: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Zero-shot genre tagging — dot product of `emb` against cached genre text embeddings.

    Returns a list of (label, prob) sorted desc, length min(k, len(GENRE_LABELS)).
    Returns an empty list if the text path failed to initialize (handled gracefully).
    """
    load()
    if _genre_text_emb is None:
        return []
    emb_np = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(emb_np))
    if n > 0:
        emb_np = emb_np / n
    sims = _genre_text_emb @ emb_np
    # Softmax over similarities for a probability-shaped output (matches CLAP semantics).
    exps = np.exp(sims - float(sims.max()))
    probs = exps / float(exps.sum())
    order = np.argsort(-probs)[: max(1, int(k))]
    return [(config.GENRE_LABELS[int(i)], float(probs[int(i)])) for i in order]


def model_id() -> str:
    """Pinned HF model identifier — surfaces via /health and on the response payload."""
    return config.AUDIO_ENCODER_MODEL_ID
