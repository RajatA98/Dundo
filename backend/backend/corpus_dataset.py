"""Resolve Phase-2 corpus artifacts from a Hugging Face Dataset repo."""

from __future__ import annotations

import os
from pathlib import Path

from backend.scripts.phase2_catalog import validate_manifest_file_hashes

CORPUS_PATTERNS = [
    "corpus.json",
    "artists.json",
    "embeddings.npy",
    "segment_embeddings.npz",
    "manifest.json",
    # examples.json + self_retrieval.json are in manifest.json's sha256 file list,
    # so validate_manifest_file_hashes() requires them present after snapshot —
    # fetch them too or validation fails on first HF-dataset load.
    "examples.json",
    "self_retrieval.json",
    "*.index",
]


def resolve_corpus_dir(default_dir: Path) -> Path:
    """Return local corpus dir, hydrating from HF Dataset when configured.

    Env:
        DUNDO_CORPUS_DATASET_REPO: e.g. ``RajatA98/dundo-corpus``.
        DUNDO_CORPUS_DATASET_REVISION: optional revision/commit.
        DUNDO_CORPUS_CACHE_DIR: optional snapshot cache dir.

    When no Dataset repo is configured, this returns ``default_dir`` unchanged.
    """
    repo_id = os.getenv("DUNDO_CORPUS_DATASET_REPO")
    if not repo_id:
        return default_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for DUNDO_CORPUS_DATASET_REPO") from exc

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=os.getenv("DUNDO_CORPUS_DATASET_REVISION") or None,
            cache_dir=os.getenv("DUNDO_CORPUS_CACHE_DIR") or None,
            allow_patterns=CORPUS_PATTERNS,
        )
    )
    validate_manifest_file_hashes(snapshot)
    return snapshot
