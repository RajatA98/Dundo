"""Backend-wide constants. Tunable here without touching analysis code."""

from __future__ import annotations

import os

# --- analysis -----------------------------------------------------------------

ANALYSIS_SR = 22050           # librosa load sample rate (mono signals)
CLAP_SR = 48000               # CLAP processor expects 48 kHz
FRAME_SIZE = 2048
HOP_SIZE = 512
EPS = 1e-7
WAVEFORM_BINS = 180           # the frontend Waveform component expects exactly 180

# --- live-upload caps ---------------------------------------------------------

# Truncate uploads before windowed MuQ encoding. Each 10 s window is a separate
# CPU forward pass, so this directly sets /neighbors latency. Default 480 s (8 min)
# effectively covers any real Suno/Udio track full-length (measured ~1.14s/window,
# so a ~3 min song's ~17 windows costs ~19s encode / ~28s end-to-end on the Space —
# see the raised client timeout in api.js) while still bounding a pathological
# upload; MAX_UPLOAD_BYTES below is the other guard. Env-tunable via CLIP_CAP_S.
CLIP_CAP_S = int(os.getenv("CLIP_CAP_S", "480"))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# MIR (tempo/key/chroma/MFCC via librosa) is stable well before full-song length,
# so cap the MIR analysis window separately from the (now full-song) MuQ encode —
# running librosa over an 8-minute buffer would add latency for no matching gain.
MIR_ANALYSIS_MAX_S = int(os.getenv("MIR_ANALYSIS_MAX_S", "60"))

# In-process narrative cache size (LRU). Repeat /narrative calls for an identical
# context+mode short-circuit the GPT-4o-mini call. Env-tunable.
NARRATIVE_CACHE_MAX_ENTRIES = int(os.getenv("NARRATIVE_CACHE_MAX_ENTRIES", "512"))

# Per-client rate limit for /narrative (the one endpoint with a real per-call
# dollar cost). Sliding window; keyed on the first X-Forwarded-For hop. Read at
# request time (not import-cached) so tests can monkeypatch. Env-tunable.
NARRATIVE_RATE_LIMIT_MAX = int(os.getenv("NARRATIVE_RATE_LIMIT_MAX", "20"))
NARRATIVE_RATE_LIMIT_WINDOW_S = int(os.getenv("NARRATIVE_RATE_LIMIT_WINDOW_S", "60"))
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
ALLOWED_MIME_PREFIX = "audio/"

# --- audio encoder (ADR-0002: swap LAION-CLAP → MuQ-MuLan) -------------------
#
# MuQ-MuLan is a CLIP-style music-text joint embedder (~700M params) from
# Tencent AI Lab, January 2025 SOTA on MagnaTagATune zero-shot tagging.
# We use it for both similarity (audio path) and zero-shot genre tagging
# (text path), so a single model load powers both jobs.
#
# Sample rate: 24 kHz (LAION-CLAP was 48 kHz). The encoder resamples inputs
# internally so callers can pass any sr — but the runtime cost favors sending
# audio at the target rate when possible.

AUDIO_ENCODER_MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
AUDIO_ENCODER_SAMPLE_RATE = 24000
# Joint embedding dim. MuQ-MuLan paper §3.3 specifies 512-d shared space —
# same as LAION-CLAP, so the catalog matrix shape is unchanged across the swap.
AUDIO_ENCODER_EMBED_DIM = 512

# Backward-compat aliases — older code references CLAP_*; keep them mapped to
# the new constants until a follow-up rename PR migrates references.
CLAP_MODEL_ID = AUDIO_ENCODER_MODEL_ID
CLAP_EMBED_DIM = AUDIO_ENCODER_EMBED_DIM
CLAP_SR = AUDIO_ENCODER_SAMPLE_RATE
CLAP_GENRE_TOP_K = 3
CLAP_GENRE_TEMPERATURE = 10.0

# --- windowed encoding (Phase 1 + 2) -----------------------------------------

# 10 s windows of audio fed independently to CLAP, then mean-pooled and L2-
# normalized to produce a single track-level embedding. Matches PROJECT_PLAN
# Phase 1 + 2 acceptance criteria and LOCKED_DECISIONS track-length protocol.
CLAP_WINDOW_SECONDS = 10
# Soft cap on the per-window count; tied to CLIP_CAP_S above, so a full-length
# (up to 480 s) query produces at most ~48 windows. Catalog inputs (30 s previews)
# produce 1-3.
CLAP_QUERY_MAX_SECONDS = CLIP_CAP_S
CLAP_POOLING = "l2_normalized_mean"

# Provisional "Completely unique" cutoff carried over from prior project — NO
# published CLAP-512 threshold data exists. Recalibrate from negatives
# distribution after the golden set is built. See PRESEARCH Q1.
SIMILARITY_THRESHOLD_DEFAULT = 0.70

GENRE_LABELS: list[str] = [
    "Synthwave",
    "Lo-fi Hip-Hop",
    "Ambient",
    "Trap",
    "Indie Folk",
    "House",
    "Drum & Bass",
    "Cinematic",
    "Industrial",
    "Jazz Fusion",
    "Phonk",
    "Orchestral",
    "Hyperpop",
    "Dream Pop",
    "Techno",
]
GENRE_PROMPT_TEMPLATE = "a {label} music track"
