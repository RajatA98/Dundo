# Dundo — Project Overview

**A content-based music similarity scanner targeting AI-generated tracks. Live, public, single-developer build.**

This document is a comprehensive snapshot for an external reader (LLM or human) who wants to understand what's been built, why, and what's still open. Written 2026-06-12, halfway through the deploy/polish phase.

---

## 1. What it is

**One-line product description**: Upload an AI-generated music track (typically a Suno or Udio output). Get back the top-3 closest real songs from a hand-curated reference catalog, ranked by acoustic similarity percentage. If nothing crosses a confidence threshold, the headline reads "Completely unique — this track doesn't sound like anything in our reference catalog."

Two independent secondary signals sit on the same report card:
- **ACRCloud Cover Song ID**: does this resemble a known commercial composition?
- **ACRCloud AI Music Detector**: is this AI-generated, and probabilistically from which engine?

A separate `/evaluation` page reports measured retrieval quality: Recall@1, Recall@3, MRR, top-1 cosine distribution on unrelated negatives, latency percentiles, and a named-examples section (currently empty, scaffolded for Option B Suno-curated examples).

**Live demo**: https://dundo-xi.vercel.app
**Backend**: https://rajata98-dundo.hf.space (FastAPI on HF Spaces CPU Basic, free tier)
**Source**: https://github.com/RajatA98/Dundo

---

## 2. Why it exists

**Honest framing**: Dundo is a job-hunt artifact aimed at a warm-intro pitch to Suno's Head of Engineering. The technical premise is sincere — content-based music retrieval against an AI generation is a real and tractable problem — but the build was scoped to demonstrate engineering judgment, not to ship a consumer product.

The name is a nod to Silicon Valley (the HBO show). In the pilot episode, Richard Hendricks pitches Pied Piper as a tool for songwriters to search whether their melody resembles anything that's come before. Investors laugh, the show pivots to compression. Dundo-the-project is Richard's original pitch, ten years later, applied to AI-generated music. The engineering is straight; the framing is a wink.

**Strategic motivation**: Suno was sued by the RIAA in 2024 over training-data concerns. Even setting aside the legal outcome, Suno almost certainly needs an internal "originality check" — every generation passing through an embedder, ANN-searched against a reference catalog, flagged when too close to existing copyrighted material. Dundo is the prototype of that internal feature. The warm-intro pitch leverages this mapping.

---

## 3. Architecture

```
User → Vercel (React/Vite frontend) → HF Space (FastAPI backend) → returns JSON
                                          │
                                          ├─ /analyze  → 7-signal librosa quality report (legacy from earlier project)
                                          └─ /neighbors → top-K acoustic similarity matches
                                                           │
                                                           ├─ LAION-CLAP music-tuned encoder (revision pinned)
                                                           ├─ 10s windowed L2-normalized mean-pool
                                                           ├─ Cosine sweep against catalog (NumPy, sub-millisecond)
                                                           └─ Optional ACRCloud Cover Song ID + AI Music Detector
```

**Stack**:
- **Backend**: Python 3.11, FastAPI, librosa, soundfile, transformers, torch (CPU wheels)
- **Frontend**: React 18, Vite 5, Tailwind v4, Framer Motion, React Router
- **Embedder**: `laion/larger_clap_music`, pinned to commit `a0b4534a14f58e20944452dff00a22a06ce629d1`
- **Backend host**: Hugging Face Space, Docker SDK, CPU Basic (2 vCPU, 16 GB RAM, free tier)
- **Frontend host**: Vercel free tier
- **Observability**: Sentry stubs wired in both ends, gated on DSN env var (currently dormant)
- **CI**: GitHub Actions (backend pytest, frontend Vitest + build + bundle-leak check, eval reproducibility check)

---

## 4. Embedding protocol

This is the part that determines retrieval quality, so it's documented carefully.

- **Audio decode**: librosa, mono at 22.05 kHz (config.ANALYSIS_SR)
- **Query length cap**: 90 seconds (config.CLIP_CAP_S)
- **Windowing**: non-overlapping 10-second windows; trailing window kept if >= 50% of window length
- **Per-window encoding**: CLAP forward pass on each window → 512-d float32 vector
- **L2-normalize each window vector**
- **Mean-pool across windows** → L2-normalize the mean → final track-level embedding
- **Also return per-window vectors** so the backend can compute both `meanPooledSimilarity` (the ranking signal) and `maxSegmentSimilarity` (local resemblance secondary signal)

**Cosine similarity formula at retrieval time**: NumPy dot product over L2-normalized vectors. No FAISS yet — catalog is 160 tracks so cosine sweep is sub-millisecond. FAISS becomes worthwhile around 50K tracks.

---

## 5. Catalog

**Current size**: 160 tracks, two-tier:

| Tier | Source | Count | Status |
|---|---|---|---|
| Tier 1 | iTunes Search API previews (30s m4a, streamed not cached per Apple terms) | 10 | Real names (Blinding Lights, bad guy, Old Town Road, etc.) |
| Tier 2 | MTG-Jamendo research dataset (CC-licensed indie artists) | 150 | Anonymized names per academic distribution convention; pending enrichment via Jamendo's free dev API |

**Storage shape**:
- `corpus.json` — track metadata + IDs
- `embeddings.npy` — (160, 512) float32 mean-pooled vectors, L2-normalized
- `segment_embeddings.npz` — per-window vectors keyed by track ID (3,495 total segments across 160 tracks)
- `manifest.json` — model SHA, threshold default (0.70), generated_at, embedding_dim
- `examples.json` — staged precomputed query responses for the home page chips

**All shipped in-container on the HF Space** (~7 MB total) so `/neighbors` works from cold start with no separate volume mount.

**Catalog growth plan** (not yet executed):
- Day 1 (now): Jamendo enrichment → real names + audio playback URLs for the 150 Tier-2 tracks
- Day 2: iTunes expansion 10 → 1,500 tracks via Search API crawl (Billboard/genre charts); biased 3:1 toward iTunes because indie unknowns don't deliver the "wow, that's recognizable" reaction
- Day 3 (optional): Jamendo expansion 150 → ~1,500

---

## 6. Evaluation

The hard claim of any retrieval system is "what's the recall?" Dundo ships a real eval, not a confusion matrix.

**Methodology**: Leave-one-out (LOO) over the live 160-track catalog. For each track, the track is held out of the index; the remaining 159 are queried using the held-out track's CLAP embedding; the held-out track's rank in the returned top-K is recorded. This is a catalog **retrieval check**, NOT an end-to-end AI-soundalike test — the methodology paragraph names this trade-off explicitly.

**Current numbers** (from `quality-scorer/public/corpus/eval.json`):
- **Recall@1**: 0.394
- **Recall@3**: 0.494
- **MRR**: 0.458
- **n_queries**: 160
- **Latency p50**: 0.28 ms (per `/neighbors` ranking call against in-memory catalog; excludes decode + CLAP encode time, which are bounded by file size not index size)
- **Latency p95**: ~0.4 ms
- **Latency p99**: ~0.5 ms

**Reproducibility loop**: a GitHub Actions workflow re-runs `python -m backend.scripts.run_eval` on every PR that touches eval inputs or the corpus and fails the build if regenerated `eval.json` differs from committed (ignoring `manifest.generated_at`). Audit-grade.

**Known limitation acknowledged in the methodology prose**: with only 10 iTunes tracks against 150 Jamendo tracks, the LOO recall is inflated by the per-artist clustering inside Jamendo (multiple tracks by `artist_005716` are close in embedding space, so a held-out track from that artist often retrieves another track by the same artist at rank 1, which counts as a "hit").

---

## 7. The genuine problem we hit during deploy

When we tested the live system with a real Suno track ("Blacktop Halo-2", a Phonk generation), the report rendered as "100% similar / 100% / 100%" across the top 3 — all three matches were Jamendo tracks. The user's reaction was "100 is crazy."

Diagnosis:
- The cosines were genuinely 0.998 / 0.997 / 0.996 — the math was correct
- The frontend used `Math.round(cosine * 100)` so all three rounded to 100
- The deeper cause: **LAION-CLAP music-tuned embeddings cluster very tightly**. Any two music tracks score 0.93-1.00 cosine against each other. CLAP collapses everything into a "music" region of 512-dimensional space.

**Fix shipped**: changed rounding to 1-decimal precision (`Math.round(cosine * 1000) / 10`) so close values stay distinguishable on the badge. Now renders 99.8% / 99.7% / 99.6%.

**Deeper fix not shipped**: a rescaled display that maps the typical [0.93, 1.00] cluster range to [0%, 100%] for visible spread. This would change the interpretation of the percentage (it would no longer be "raw cosine") but would make the rankings actually communicate. Open question whether to ship this.

---

## 8. What's live and verified

| Component | State |
|---|---|
| GitHub repo (public) | Live: https://github.com/RajatA98/Dundo |
| HF Space backend | Live, `/health` returns corpus:160, segments:3495 |
| Vercel frontend | Live: https://dundo-xi.vercel.app |
| CORS wiring (CORS_ORIGIN → Vercel URL) | Confirmed via preflight test |
| GitHub → Vercel auto-deploys | Wired and verified |
| Sentry stubs (backend + frontend) | In code, gated on DSN env vars (currently no-op) |
| 1-decimal similarity display | Live |
| Audio playback for iTunes Tier-1 tracks | Just shipped |
| GitHub Actions CI (3 workflows) | Live |

## 9. What's pending

| Item | Estimated effort | Blocked on |
|---|---|---|
| Jamendo enrichment (real names + audio URLs) | 2-3 hours | User signing up at devportal.jamendo.com for Client ID |
| Catalog expansion to ~3,000 tracks | 1-2 days | Sign-up above + decision to proceed |
| UptimeRobot keepalive ping on `/health` | 3 min | User signup (just needs them to add the monitor) |
| ACRCloud trial activation + secrets | 15 min | Time-sensitive: 14-day trial, should align with warm-intro send |
| Sentry account + DSN paste | 10 min total | User signup |
| Vercel Analytics enable | 30 sec | One checkbox in dashboard |
| Custom domain (optional) | 10 min + $12/yr | Decision whether to bother |

---

## 10. Open decisions / questions worth thinking through

These are the things I'm actively weighing or have explicitly not decided:

1. **Display percentage rescaling.** Should we ship the [0.93, 1.00] → [0%, 100%] rescale so the percentage genuinely communicates ranking? Or keep the raw 1-decimal cosine to preserve "I'm showing you the real number" honesty? The trade is interpretability vs honesty.

2. **Catalog composition strategy.** Current bias toward Jamendo (94%) is wrong for an audience that expects to recognize matches. Day 2 plan rebalances to 50/50 iTunes/Jamendo. But at scale (10K+ tracks), iTunes Search API becomes rate-limited; need to decide whether to invest in robust crawl tooling or accept ~5K iTunes as the upper bound.

3. **Spotify integration**. Their Web API was largely closed November 2024 (audio-features, audio-analysis, recommendations all returning 403 to new apps; still unrestored 18 months later). Conclusion has been "don't bother." But is there an angle I'm missing?

4. **Custom encoder fine-tune for AI-generated music**. The catalog's embedding space is clustered tightly because LAION-CLAP wasn't trained discriminatively on AI-generated content. Fine-tuning on a paired (Suno generation, source-of-inspiration commercial track) dataset would spread the embedding distribution and improve retrieval — but requires the Suno-targeted golden set we've deferred (Option B). Worth doing pre-warm-intro?

5. **Hum-to-AI-generation extension**. Bigger product idea: hum into a microphone, identify the closest reference track in the catalog, feed the matched track's metadata as a prompt to Suno's API, return a generated track in that style. This is genuinely a product gap (Google Hum to Search → identify; Suno → generate; nobody has combined them). Could be a follow-up project, or could be the actual Day-1 pitch to a Suno engineer ("here's what I'd build at Suno on Day 1, and I have most of the retrieval half already shipped").

6. **Name + visual identity.** Considered (and rejected) renaming to something Suno-themed. Conclusion: the Silicon Valley wink is the strongest differentiator. Visual identity is cream (#FAFAF7) + ink (#0E1620) + green accent (#0FAA60), with a single reserved color (#F25C54) for the "likely Suno" pill on the AI Music Detector row. Open question whether to add a small wordmark or keep type-only.

---

## 11. What this project is NOT

- **Not RAG.** No LLM in the loop, no generation step. Pure vector retrieval / semantic search. The output IS the search results, returned as JSON.
- **Not a Shazam clone.** Shazam does exact-fingerprint matching against ~30M licensed commercial tracks. Dundo does semantic similarity against 160 tracks and is bounded by what a solo developer can legitimately license (i.e., not much).
- **Not production-grade.** $0 hosting, no SLA, free tier, single CPU. Cold-start is ~30 seconds; warm response is sub-second.
- **Not generative.** It does not create music. It only finds existing reference music similar to an upload.

---

## 12. If you're an LLM reading this

Helpful framings to think about:

- What's the strongest critique of the eval methodology that I'm not already aware of?
- Given the embedding-clustering problem (CLAP putting all music in a tight cone), is the display rescale the right fix, or is there a better one?
- The hum-to-AI-generation extension — does that map to a real product gap, or is it crowded?
- The Suno warm-intro framing — is there a sharper way to position this for someone at Suno?
- Anything obviously missing from the product story that would change a senior engineer's read of it?

Feel free to push back, find weak points, propose alternative angles. The project benefits from skeptical readers more than supportive ones at this stage.
