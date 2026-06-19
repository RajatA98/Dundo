"""CLAP smoke test — marked `slow` so the default `pytest` run doesn't trigger
the ~2 GB checkpoint download. Opt in with `pytest -m slow`.

The first run takes several minutes (download). Subsequent runs are <30 s once
the model is cached under `~/.cache/huggingface/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import CLAP_EMBED_DIM, GENRE_LABELS


@pytest.mark.slow
def test_clap_load_encode_smoke() -> None:
    from backend.clap_engine import encode_audio, load, top_genres

    load()
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    emb = encode_audio(sine, sr)
    assert emb.shape == (CLAP_EMBED_DIM,)
    assert emb.dtype == np.float32
    # L2-normalized → norm is 1 (within float precision).
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4

    genres = top_genres(emb, k=3)
    assert len(genres) == 3
    for label, prob in genres:
        assert label in GENRE_LABELS
        assert 0.0 <= prob <= 1.0
    # Probabilities are a softmax; total mass ≤ 1.
    total = sum(p for _, p in genres)
    assert 0.0 < total <= 1.0
