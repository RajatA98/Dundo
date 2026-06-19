# ADR-0002: Swap LAION-CLAP for MuQ-MuLan as PiedPiper's audio encoder

**Status**: Accepted
**Date**: 2026-06-12
**Decider**: Rajat Arora (after live diagnostic with Claude-Code + Perplexity)

---

## Context

PiedPiper currently uses LAION-CLAP (specifically `laion/larger_clap_music`, pinned to commit `a0b4534a14f58e20944452dff00a22a06ce629d1`) as the audio encoder behind every similarity match. Shipped with this stack:

- A 160-track reference catalog with 512-dim L2-normalized embeddings
- LOO retrieval eval at R@1=0.394, R@3=0.494, MRR=0.458
- ADR-0001 calibration to handle the cosine-clustering ("anisotropy") problem at the presentation layer

The presentation-layer calibration shipped successfully but did not fix the underlying problem: in the live demo, an unrelated Suno Phonk generation scored 99.8% / 99.7% / 99.7% raw cosine against three random Jamendo tracks, and ACRCloud's Cover Song ID independently confirmed those tracks were not actually similar compositions. The empirical pairwise-cosine distribution across the catalog (collected during ADR-0001) showed 80% of unrelated pairs sitting in [0.95, 1.00] — confirming the model itself is collapsing all music into a narrow cone of embedding space.

A research sweep across the 2024-2026 audio embedding literature (saved at `~/Documents/CLAP_Industry_Adoption_Better_Alternatives_20260612/CLAP_vs_Real_World.md`) returned three findings that drove this decision:

1. **Almost no production music ML company publicly ships LAION-CLAP for similarity.** It is used by Stability AI and Facebook AI Research as an *evaluation metric* for generation models, not as a retrieval backbone. Spotify, Suno, Udio, Google MusicLM, Cyanite, ACRCloud, and the major-label R&D teams either ship proprietary embedders or use fundamentally different math (fingerprinting, collaborative filtering).
2. **MuQ-MuLan (Tencent AI Lab, arXiv 2501.01108, January 2025) is the new open-source SOTA.** It holds state-of-the-art on MagnaTagATune zero-shot tagging, beating LAION-CLAP on both ROC-AUC and PR-AUC. Open checkpoint at `OpenMuQ/MuQ-MuLan-large` under CC-BY-NC 4.0. ~700M parameters.
3. **A separately fact-checked competitive landscape sweep** (saved at `~/Documents/CLAP_Industry_Adoption_Better_Alternatives_20260612/`) revealed that the closest direct competitors are **SoundPatrol** (UMG + Sony endorsed, label-side, closed enterprise) and **Soundverse Trace** (vertically locked to their own generator). PiedPiper's actual position — creator-side, generator-agnostic, open and explainable, with measurable LOO eval — occupies a real product gap. Shipping LAION-CLAP behind that position is a weak technical claim ("I used the research baseline"); shipping a 2025 SOTA encoder + custom calibration + measurable improvement is materially stronger.

---

## Decision

Replace `laion/larger_clap_music` with `OpenMuQ/MuQ-MuLan-large` as the primary audio encoder in the PiedPiper backend. Re-encode the entire 160-track catalog. Re-run the LOO eval. Publish the before/after numbers in the README's evaluation section.

### What changes

| Layer | LAION-CLAP (before) | MuQ-MuLan (after) |
|---|---|---|
| Python package | `transformers` (already a dep) | `muq` (new pip dep) |
| Model class | `ClapModel.from_pretrained(...)` | `MuQMuLan.from_pretrained(...)` |
| HF model ID | `laion/larger_clap_music` | `OpenMuQ/MuQ-MuLan-large` |
| Parameter count | ~150M | ~700M |
| Audio sample rate | 48,000 Hz | 24,000 Hz |
| Embedding dim | 512 | 512 (confirmed from MuQ paper §3.3) |
| Genre tagging | `top_genres()` via cached text embeddings | Preserved — MuQ-MuLan is CLIP-style and supports the same text-prompt pattern |
| L2-normalized output? | Yes | Yes (consistent with MuLan-style CLIP-derivative architectures) |

### What stays the same

- The 10-second windowed mean-pool protocol (locked from PROJECT_PLAN Phase 2). MuQ-MuLan is a different model but our windowing strategy is encoder-agnostic.
- The catalog ingest pipeline (`rebuild_corpus.py`) — only the inner encode call changes.
- The similarity math in `similarity.py` (`top_k_neighbors`, FlatCatalog, cosine sweep). Vectors are still 512-dim and L2-normalized.
- The calibration logic in ADR-0001 (percentile rank, similarity label, querySpecificity). The calibration depends on catalog distribution, not the encoder identity — it auto-adapts to whichever encoder produces the catalog.
- The wire shape of `/neighbors`. Backward-compatible.
- The 160-track catalog metadata (titles, artists, audio URLs, artwork). Only the embeddings change.
- The LOO eval methodology. Numbers will change; methodology won't.
- The HF Space + Vercel deploy topology.

### What gets re-measured

After the swap + catalog re-encode:

- **LOO eval**: R@1, R@3, MRR — should improve given MuQ-MuLan's SOTA tagging numbers, but is not guaranteed.
- **Pairwise cosine distribution**: mean, std, percentile floor. The anisotropy diagnostic (mean of random pairs) is the headline number.
- **Discrimination ratio**: `cosine(query, top_match) - cosine(query, random_catalog_track)`. Should improve.
- **Encode latency per windowed track**: MuQ-MuLan is ~5× larger than LAION-CLAP; CPU encode time will be 2-5× slower. Acceptable cost if discrimination materially improves.

---

## How (implementation plan)

### Phase 1 — Code changes

1. **Add `muq` to `backend/pyproject.toml` `[runtime]`** alongside `transformers` (which remains for inherited components).
2. **Create `backend/backend/muq_engine.py`** — replaces `clap_engine.py` for the encoder path. Same public interface: `load()`, `encode_audio(wav_mono, sr)`, `top_genres(emb, k=3)`. Internal: loads MuQ-MuLan, resamples input to 24kHz, returns L2-normalized 512-d embedding.
3. **Refactor `clap_windowed.py`** to call the new module. Module name kept (for now) to avoid widening the diff; internally it just calls `muq_engine.encode_audio` instead of `clap_engine.encode_audio`. A rename to `audio_windowed.py` can be a follow-up cleanup.
4. **Update `config.py`** with new model ID + sample rate + pinned revision (whatever the current MuQ-MuLan release commit SHA is on HF).
5. **Update `deploy/hf_space/Dockerfile`** — install `muq` package, pre-pull MuQ-MuLan weights at build time so the Space cold-start doesn't have to download ~2.8 GB on first request.
6. **Update `manifest.json`** writer in `rebuild_corpus.py` to stamp the new model SHA.

### Phase 2 — Catalog re-encode

Run `python -m backend.scripts.rebuild_corpus` end-to-end:
- Re-fetches iTunes previews + Jamendo audio from source URLs (no cached audio on disk).
- Runs each track through windowed MuQ-MuLan encoding.
- Writes new `embeddings.npy`, `segment_embeddings.npz`, `corpus.json` (preserves the enriched Jamendo metadata from the prior pass), `manifest.json`.

Expected wall-clock: 45-90 minutes on the laptop CPU. Network fetches dominate the first ~30 min; MuQ-MuLan inference the rest.

### Phase 3 — Eval delta

1. Run LOO eval against the new catalog (`python -m backend.scripts.run_eval`).
2. Compare R@1, R@3, MRR, latency p50/p95/p99 against the LAION-CLAP baseline.
3. Recompute the pairwise cosine distribution; report mean, p95, p99.
4. Append both sets of numbers to `eval.json` (top-level keys `metrics` and `metrics_legacy_clap`) and to ADR-0001's empirical-evidence table.

### Phase 4 — Deploy

1. Upload corpus + new code to HF Space via `huggingface_hub`.
2. Trigger factory rebuild so the Dockerfile re-runs (new image with `muq` installed + MuQ-MuLan weights baked in).
3. Verify `/health` reports the new `model_sha`. Verify `/neighbors` returns the new distribution.

### Phase 5 — Documentation

1. Update README's "Architecture" + "Embedding protocol" sections with the new model.
2. Update README's "Evaluation" section with before/after numbers.
3. Mark this ADR's implementation tracker complete with measured deltas.
4. Update PROJECT_OVERVIEW.md.

---

## Why MuQ-MuLan specifically (alternatives considered)

### A. Keep LAION-CLAP, apply mean-centering + top-1 PC removal

The other path the earlier research surfaced. Strictly cheaper (~30 min of code, no re-encode required, no new dependency). Literature reports mean-centering alone drops unrelated-pair cosines from 0.93-0.99 down to 0.4-0.7.

**Rejected because**: it fixes the symptom inside a known-limited model. A swap to MuQ-MuLan plus calibration tackles both the symptom AND the underlying model quality. The competitive-landscape research showed that "I used CLAP" is a weak technical claim regardless of how clever the post-processing is. The pitch for Suno's Product Engineer role is materially stronger with "shipped on 2025 SOTA + measured the delta" than with "shipped on 2023 baseline + a clever calibration."

**Reconsider if**: MuQ-MuLan empirically fails to improve discrimination on the LOO eval (R@1 / discrimination-ratio do not move materially). In that case the swap is sunk cost and mean-centering goes back on the table.

### B. Swap to MERT (m-a-p)

MERT is a credible alternative — BERT-style music SSL, validated on 14 MIR tasks. A 2025 MDPI paper reports LAION-CLAP outperforms MERT on fine-grained text-aligned tasks (CLAP TEST trust 0.9164 vs MERT 0.8763 on cymbal classification), suggesting CLAP > MERT for our specific use case.

**Rejected because**: MuQ-MuLan benchmarks beat both CLAP and MERT on MagnaTagATune zero-shot, and the architecture (Mel Residual Vector Quantization + MuLan-style CLIP head) is structurally newer.

**Reconsider if**: MuQ-MuLan's CC-BY-NC license becomes a problem (e.g., if the project pivots to commercial use). MERT is permissively licensed.

### C. Swap to MAEST / OMAR-RQ / other 2024-2025 encoders

These are also competitive but with thinner published benchmark coverage than MuQ-MuLan as of June 2026.

**Rejected because**: MuQ-MuLan has the strongest publicly-replicated headline numbers and the most active upstream repo (Tencent AI Lab official, recent commits).

### D. Train a custom encoder from scratch on Suno-paired data

The MuQ training stack includes a publicly available dataset of 116,000 fully-licensed Suno V5 outputs paired with style descriptions. Genuinely tractable: fine-tune the MuQ-MuLan projection head on contrastive (Suno generation, prompt-referenced reference) pairs.

**Deferred to ADR-0003.** This is the right "second step" — first prove the off-the-shelf MuQ-MuLan swap improves PiedPiper's numbers, then fine-tune on the Suno-paired dataset to specialize. Trying to do both in one ADR creates two confounded changes that we can't attribute the delta to.

### E. Use an audio fingerprinting library (chromaprint, dejavu) instead of neural embeddings

Different tool for a different question. Fingerprinting answers "is this an exact recording of a known song?" — that's already covered by ACRCloud's Cover Song ID integration. Neural embeddings answer "what does this sound like in general acoustic feel?" — that's the question PiedPiper is built around.

**Rejected because**: not a substitute for the neural embedding; complementary signal we already ship.

### F. Keep LAION-CLAP and rely entirely on ACRCloud + a manual "this is similar in vibe" disclaimer

Cheapest option: do nothing. UX would still feel broken without ADR-0001's calibration + this swap.

**Rejected because**: PiedPiper would not be technically interesting enough to anchor a Suno Product Engineer pitch.

---

## Consequences

### Positive

- The headline pitch becomes substantively stronger: "I shipped a 2025 SOTA music encoder, compared it head-to-head with the 2023 baseline on my own catalog and the Da-TACOS public benchmark, and chose [winner] based on the data." That's an engineering decision, not a hobbyist's choice.
- The cosine distribution should genuinely spread (the model is trained with better contrastive coverage), which reduces the anisotropy disease the calibration is currently masking.
- The ADR commitment to "before and after" eval numbers turns this into a verifiable engineering claim, not a vibe.
- The architecture story becomes useful for the Suno application: "the right model for AI-music similarity in 2026 is not the 2023 research baseline; here's the data on why."

### Negative / costs

- **Catalog re-encode** — 45-90 min of CPU work the first time, plus ~30 min network. Re-runs (e.g., when expanding the catalog) hit the same cost.
- **Model footprint** — MuQ-MuLan weights are ~2.8 GB on disk vs LAION-CLAP's ~1.5 GB. HF Space Docker image grows from ~5-6 GB to ~7-8 GB. Still within HF free-tier image limits.
- **Inference latency** — 700M-param model on CPU will be 2-5× slower per encode than 150M-param CLAP. Per-query latency at /neighbors increases. Acceptable trade given the discrimination improvement expected.
- **License** — CC-BY-NC 4.0. Non-commercial use only. For a portfolio piece this is fine; for commercial deployment it would need relicensing or a swap back to a permissive model. README must call this out explicitly.
- **New pip dependency** (`muq`) — adds ~50 MB to the image, plus its own transitive deps. Stable upstream (Tencent AI Lab) so risk is low.
- **One-way decision** — once we re-encode the catalog and ship the new embeddings, going back to LAION-CLAP means another re-encode. Not free, but recoverable.

### Eval impact (measured after Phase 3 ran)

**Empirical result on the 155-track MuQ-MuLan catalog vs 160-track LAION-CLAP baseline** (5 Jamendo tracks lost to CDN failures during re-encode are not yet recovered):

| Metric | LAION-CLAP (n=160) | MuQ-MuLan (n=155) | Delta |
|---|---|---|---|
| **R@1** | **0.394** | **0.639** | **+62%** |
| **R@3** | **0.494** | **0.735** | **+49%** |
| **MRR** | **0.458** | **0.692** | **+51%** |
| Mean of random pairwise cosines | 0.967 | **0.456** | **−0.511** |
| Std of pairwise cosines | 0.030 | **0.186** | **+0.156 (6× wider spread)** |
| Min pairwise cosine | 0.694 | **−0.084** | actual negative cosines now possible |
| p1 of pairwise cosines | 0.857 | **0.105** | floor dropped 75 percentage points |
| p50 of pairwise cosines | 0.976 | **0.431** | median random pair dropped half a unit |
| p99 of pairwise cosines | 0.997 | 0.934 | top no longer saturated |
| **Discrimination ratio** (mean: cosine(q, top) − cosine(q, random)) | **0.036** | **0.451** | **12.4× wider gap** |
| /neighbors LOO ranking latency p50 | 0.28 ms | 0.27 ms | flat (NumPy cosine sweep is encoder-agnostic) |
| /neighbors LOO ranking latency p95 | ~0.4 ms | 0.74 ms | slight regression, still microseconds |

**Read of the deltas:**

- **R@1 +62%** is a substantial retrieval-quality improvement. MuQ-MuLan correctly returns the held-out track at rank 1 for 64% of LOO queries vs 39% for LAION-CLAP. The model is genuinely better at audio retrieval on this catalog.
- **The discrimination ratio improvement (0.036 → 0.451) is the most damning number for the LAION-CLAP baseline.** CLAP could barely distinguish a true match from a random catalog track (0.036 cosine separation); MuQ-MuLan separates them by 12.4×. This is the *empirical reason* why CLAP-based displays were forced to show "100% / 100% / 100%" — there was no actual signal to differentiate the top from the rest.
- **The pairwise cosine distribution properties are textbook anisotropy fix.** Mean drops half a unit, std 6× wider, the floor of the distribution becomes actually negative (some catalog pairs are genuinely dissimilar). The literature predicted this; the empirical result confirms it dramatically.
- **Encode latency was not measured per query in this pass** — only LOO ranking latency (which is encoder-agnostic NumPy work). MuQ-MuLan inference per 10-second window is ~0.8 s on CPU (vs LAION-CLAP's ~0.2 s) so the per-query end-to-end /neighbors response time will increase by ~3-5 seconds for a 90-second upload. This is the cost the ADR called out and is being absorbed.

Catalog recovery note: the rebuild ended with 155 of 160 tracks (10 iTunes + 145 of 150 Jamendo). 5 Jamendo tracks dropped during re-encode due to Jamendo CDN connection resets; they are recoverable via a follow-up rerun and don't affect the headline conclusions.

### Verification (added 2026-06-14)

Eval numbers above measure the catalog's internal consistency, not whether the matching pipeline is doing what it claims at the audio-decode → encode → retrieve level. The companion verification script `backend/backend/scripts/verify_matching.py` answers that more directly: for each iTunes Tier-1 track, it downloads the Apple preview audio fresh, POSTs it back to `/neighbors`, and checks (a) whether the same track is returned at rank 1, (b) what the self-match cosine is, and (c) whether the returned `matchTimestamp` points the strongest segment back at itself.

Measured run on 2026-06-14 against the live HF Space (model SHA `2e01c796b71dca71b45251384c04cd7b237c9020`):

| Pass criterion | Result | Threshold |
|---|---|---|
| Self at rank 1 | **10 / 10  (100%)** | ≥ 9 / 10 |
| Self-match cosine mean | **0.9988** | ≥ 0.92 |
| Self-match cosine min | 0.9925 | — |
| Timestamp aligns to self | **10 / 10  (100%)** | ≥ 8 / 10 |
| Round-trips that completed | 10 / 10 | — |

What this proves:

- The encoder is **deterministic up to numerical noise across encodes**: re-encoding the same audio at request time (fresh download → librosa decode → 24 kHz resample → MuQ-MuLan forward pass) reproduces the precomputed catalog embedding closely enough that cosine collapses to 1.0 within ~0.008. If the call signature were wrong (e.g., wrong sample rate, wrong dtype, wrong windowing convention), this number would not stick near 1.0.
- The **retrieval pipeline does what it claims**: a query's correct nearest neighbor in the catalog is the one mathematically identical to it, not some genre-cluster random.
- The **timestamp pipeline is wired correctly**: when a query is identical to a catalog track, the argmax over the per-window cosine matrix returns the same window of the same track on both sides. A bug like "off-by-one window index" or "argmax confused query/catalog axis" would have shown up here.

What this does NOT prove:

- It does not prove cross-track matches are semantically right. A track returning "very close — Blinding Lights" for a Suno generation is not validated by self-retrieval; that's a different claim and needs a different eval (Da-TACOS or SHS100K cover-song pairs, queued in ADR-0003).
- It does not prove the model is well-calibrated for distinguishing copies vs inspirations. The harness intentionally feeds identical audio; novel inputs are out of scope.

To rerun: `python -m backend.scripts.verify_matching --base-url https://rajata98-piedpiper.hf.space`. Single-target dry-run: `python -m backend.scripts.verify_matching --target tier1:itunes:1488408568`.

Side finding: implementing the harness uncovered a `.m4a` / AAC-LC decode bug — the backend's `librosa.load(io.BytesIO(...))` couldn't decode iTunes preview format because libsndfile lacks AAC support and audioread's ffmpeg fallback needs a file path. Fixed by adding a temp-file fallback in `_decode_and_pipeline` (`backend/backend/api.py`) and regression-tested in `backend/tests/test_api.py::test_decode_uses_temp_file_fallback_for_aac_lc`. The bug never affected the user's Suno mp3 uploads — only iTunes preview uploads and other AAC-formatted audio. Worth documenting because the verification work surfaced it.

### Documentation impact

- README's "Embedding protocol" + "Architecture" + "Evaluation" sections need updates.
- PROJECT_OVERVIEW.md "Architecture" + "Catalog" sections need updates.
- The CC-BY-NC license caveat needs to land somewhere visible in the README.

---

## Implementation tracker

- [x] ADR drafted and accepted.
- [x] `muq` added to `backend/pyproject.toml` `[runtime]`.
- [x] `muq_engine.py` written.
- [x] `clap_windowed.py` refactored to call MuQ.
- [x] `config.py` updated with new model ID + sample rate + revision SHA (`2e01c796b71dca71b45251384c04cd7b237c9020`).
- [x] `deploy/hf_space/Dockerfile` updated to pre-pull MuQ-MuLan weights.
- [x] `rebuild_corpus.py` needed no changes — the encoder swap propagated through `clap_windowed` cleanly.
- [x] Catalog re-encode complete (155 of 160 tracks; 5 Jamendo CDN losses queued for follow-up).
- [x] LOO eval re-run; before/after numbers captured in `eval.json` and the table above.
- [ ] HF Space deploy + verification.
- [ ] README + PROJECT_OVERVIEW.md updated.

(Implementation landed in commits referenced by `git log --grep=ADR-0002`. The deploy + README update happen in the same commit cluster.)
