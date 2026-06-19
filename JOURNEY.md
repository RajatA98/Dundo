# JOURNEY — How Dundo came to be

This is the chronological story of the decisions that led to Dundo, written for the project's author (you) so that any future session — yours or a new Claude session — picks up the *why* alongside the *what*. The factual record lives in the ADRs (`docs/decisions/`), the `_PREPIVOT/` archive (`factory/artifacts/_PREPIVOT/`), and PiedPiper's git history. This document is the *narrative* — the through-line.

If you only have five minutes, skim the **section headers**. If you have twenty, read the body.

---

## Phase 0 — The seed (early June 2026)

The original project was called **PiedPiper**, a nod to the *Silicon Valley* pilot where Richard Hendricks first pitches Pied Piper as a music-search tool for songwriters. The original framing: an *acoustic-similarity scanner for AI-generated music*. The product surface: upload a Suno generation, get a verdict on how close it lands to any track in a reference catalog, with a "match risk" headline. The audience target: Suno's leadership, with the implicit pitch *"here's a quality gate you'd want running on every generation."*

The early engineering moved fast. By mid-June PiedPiper had:
- A working FastAPI backend with `/analyze` (7-signal quality check) and `/neighbors` (top-K cosine retrieval)
- A 160-track reference catalog (mostly iTunes Search API Tier-1 hits + MTG-Jamendo Tier-2 CC tracks)
- A React + Vite frontend on Vercel
- A Docker-SDK Hugging Face Space backend
- Two ACRCloud signals (Cover Song ID + AI Music Detector) running in parallel for commercial-second-opinion verification

## Phase 1 — The CLAP problem (ADR-0001 then ADR-0002)

The live demo surfaced an embarrassing failure: every match displayed at "100% / 100% / 100%" similarity, regardless of how close the underlying audio actually was. Root-cause analysis: **contrastive-encoder anisotropy**. The pairwise cosine distribution across the 160-track catalog clustered tightly (mean 0.967, std 0.030, top-vs-random discrimination ratio 0.036). The UI was forced to round all distinct matches to the same headline number — the model could only distinguish a real match from a random catalog track by ~0.036 cosine.

The first fix (**ADR-0001**) was a calibration layer: convert raw cosines to percentile rank within the catalog's own distribution, attach coarse labels (very close / close / moderate / weak), surface a `querySpecificity` score for "broadly similar to everything." This worked as a presentation safety net but didn't fix the underlying math.

The real fix (**ADR-0002**) was swapping the encoder. After researching the 2024-2026 audio embedding literature, we replaced LAION-CLAP with MuQ-MuLan (Tencent AI Lab, Jan 2025 SOTA on MagnaTagATune zero-shot). The measured result on the full catalog: **Recall@1 +62% (0.394 → 0.639), discrimination ratio 12× wider (0.036 → 0.451), mean random-pair cosine dropped from 0.967 to 0.456.** Both encoders' numbers stay preserved in ADR-0002 so the decision remains auditable. The shipping calibration (ADR-0001) was kept as a safety net on top of the deeper math fix.

A side-effect of the swap: MuQ-MuLan is CC-BY-NC 4.0 (non-commercial portfolio use), which became the recurring constraint that eventually forced the broader pivot.

## Phase 2 — The scaling argument (ADR-0003) and decomposed similarity (ADR-0004)

**ADR-0003** argued that the percentile-rank approach survives catalog scale (density-relative calibration) — *argued, not proven at scale*. ADR-0003 was honest about that.

**ADR-0004** addressed a different criticism: one cosine doesn't defend "similar." The user (and a senior reviewer the user pictured) would reasonably ask: *similar how?* So we added four classical MIR criteria — tempo (`librosa.beat.beat_track`), key + mode (Krumhansl-Schmuckler over chroma_cens), harmonic content (chroma_cens mean cosine), timbre (MFCC mean+std cosine). Each criterion got a per-neighbor agreement score and a user-facing label ("same key," "4 BPM apart," "similar production feel"). Math is librosa-native, no new dependencies. The criteria layer is additive — top-K ordering still comes from MuQ-MuLan cosine.

The expanded-row UI got side-by-side snippet players (the matched 10-second window from both the upload and the catalog track) and the criteria comparison table.

## Phase 3 — The RAG narrative layer (ADR-0005)

The system now had retrieval (R), but the **G of RAG was missing** — no LLM-generated explanation of *why* a match was a match, no actionable creator feedback grounded in the criteria. ADR-0005 added it:

- `backend/backend/rag_narrative.py` — Pydantic models, OpenAI GPT-4o-mini client, a single `_call_openai_json` adapter, structured-citation validation against the supplied context (a hallucinated tempo or wrong-track citation surfaces as `NarrativeUnavailable`, never as text), a context-completeness gate (cookbook-named self-evaluation pattern, borrowed downward from Agentic RAG as a cost guardrail), canonical SHA-256 cache key with prompt-template + criteria-algorithm versioning, no retry loops, 8 KB prompt cap.
- `backend/backend/context_token.py` — HMAC-signed opaque token that `/neighbors` issues, carrying the per-neighbor NarrativeContext fragments + queryFingerprint + model SHA + catalog SHA + 30-minute expiry. `/narrative` accepts `{contextToken, trackId, mode}`, verifies the token, rebuilds context server-side. Stateless across HF Space restarts + workers — replaces an in-memory cache pattern that broke under load.
- `backend/backend/narrative_telemetry.py` — in-process counters surfaced at `GET /narrative/stats`.
- Three-tab UI: "Why these are similar" + "Make mine more distinctive" + "Visual match" (WaveSurfer.js Spectrogram plugin, lazy-imported on tab click).
- 12-case RAG eval harness gating five baseline metrics at 1.0 (happy-path kind agreement, low-context gate, hallucination rejection, malformed-output rejection, OpenAI-error handling).

This was the biggest engineering chunk and the most cookbook-shaped work. Codex reviewed two rounds before code, then the implementation, then a follow-up tightening pass. The rag-cookbook's central rule — *refuse to climb without measured evidence* — became the discipline for the rest of the project: Naive RAG at retrieval, metadata-grounded generation at presentation, no Hybrid / Graph / Agentic climbing until evidence demanded it.

## Phase 4 — The scaling discussion (planned ADR-0007)

By mid-June, the conversation moved to *what does this look like at 10K / 100K / 1M+ tracks?* The discussion built two new pieces of evidence:

1. **`bench_similarity.py`** — synthetic catalogs at 155 / 1K / 10K / 100K tracks measured across NumPy `means@T`, FAISS IndexFlatIP, FAISS IndexHNSWFlat. The crossover where FAISS Flat pays for itself sits around 10K tracks; HNSW is 6× faster at 100K. NumPy is genuinely the right call at PiedPiper's catalog size; the climb has trigger conditions, not preemptive complexity.
2. **`similarity_faiss.py`** — FAISS Flat backend drop-in for `top_k_neighbors`, gated by `SIMILARITY_BACKEND=faiss` env var. Same return shape, same exact cosine numbers (IndexFlatIP over L2-normalized vectors = cosine). Index built lazily, cached by `id(catalog.means)`.

These two pieces shipped to PiedPiper as commit `c138309 FAISS Flat backend + bench + SIMILARITY_BACKEND flag` on 2026-06-19, the day before the fork.

A planned ADR-0007 was going to document the full tech-stack snapshot (vector store ladder + LLM provider tier + LangFuse observability + Modal migration trigger), but the broader product pivot intervened first.

## Phase 5 — The licensing wall and the pivot (2026-06-18 to 06-19)

The recurring blocker across Phases 1–4 was the same: **commercial-catalog ingestion**. Every time we wanted to scale past 160 tracks toward a credible production catalog, we ran into "we can't legally ingest commercial big-artist catalogs." ACRCloud's 150M-track database is *their* business asset and not exposed for bulk ingestion. The iTunes Tier-1 ingest sat on a knife's edge of preview-terms compliance (ADR-0002 was honest about this). A licensed catalog like Spotify's or Apple's would require label deals well outside a portfolio project's scope.

The user reached the breaking point on 2026-06-18: *"the whole issue with this project is the licensing for the big artists."* That triggered a pivot conversation that ran across two days.

The pivot landed on a simple inversion: **stop trying to detect copyright risk against big-artist catalogs. Start helping creators discover indie artists whose work resonates with their AI generations.** The licensing wall disappears (CC + public-domain indie is bulk-ingestible). The defensive framing disappears (positive-sum discovery replaces risk verdict). The Suno-pitch alignment gets stronger (Suno's leadership cares about creator empowerment, not policing).

The new name landed in the same conversation: **Dundo** (ढूंढो), Hindi for *search* / *find*. It deliberately pairs with Suno (सुनो, Hindi for *listen*) — the search side of what Suno enables.

## Phase 6 — The reviews (2026-06-18 to 06-19)

The pivot got two independent reviews before any code moved:

- **Codex** (`factory/artifacts/CODEX_DUNDO_PIVOT_REVIEW_RESPONSE.md`): "Approve with revisions." Five ⭐ pushbacks (clone+orphan-branch not raw `cp -r`, 3 separate fork commits not one monolithic, drop iTunes Tier-1 entirely, stage catalog 5K-10K first then 50K+, port FAISS BEFORE the fork). Eight ✓ approvals. Tagline rewrite: *"Upload an AI track. Find the indie artists it resonates with."*
- **Perplexity** (chat-delivered, summarized in this session): "Qualified yes." The pivot is directionally smart and solves the two biggest problems. Main risk: audience-loop validation — do AI music creators repeatedly use a product that points them at human artists, or is it interesting once? The user chose not to pursue artist interviews (no access). Style attribution stays secondary; no "near you" language until location/show data is reliable.

Both reviews approved the direction. The plan was locked.

## Phase 7 — The fork (2026-06-19)

Per Codex's mechanics:
- **Pre-fork on PiedPiper**: commit `c138309` (FAISS) + commit `c04f38e` (`.gitignore`: factory/artifacts/ stays local going forward).
- **Local clone-fork**: `git clone --no-hardlinks PiedPiper Dundo`, orphan branch, three commits — `fork foundation` → `rename: PiedPiper → Dundo` → `retire: drop ACRCloud + iTunes Tier-1 + risk-verdict framing`.
- **Claude memory namespace copy** so the next session in `/Users/rajatarora/Projects/Dundo` has full context.
- **Artifact reorganization**: PiedPiper-era artifacts moved to `factory/artifacts/_PREPIVOT/`; pivot artifacts (PIVOT_PRD.md, CODEBASE_AUDIT.md, the Codex reviews, the Perplexity brief) stay at `factory/artifacts/` proper.
- **PiedPiper stays open** on GitHub as engineering provenance, not archived. Its README points at Dundo.
- **Tagline locked**: *"Upload an AI track. Find the indie artists it resonates with."*
- **CREDITS.md** cites PiedPiper explicitly as the foundation.
- **JOURNEY.md** (this file) captures the chronological *why*.

## What ships next

The Dundo plan from here:
1. **Push to `github.com/RajatA98/Dundo`** when the user creates the remote.
2. **Bandsintown integration** — `backend/backend/bandsintown_client.py` for per-artist show lookup, frontend "next show" row on each discovery card, `BANDSINTOWN_APP_ID` Space secret. Nullable: cards without show data just don't render that row.
3. **Strict `response_format=json_schema`** in the narrative call (currently `json_object` — the residual Pydantic-validation failures from GPT-4o-mini's looser mode are the lingering live bug).
4. **Style-attribution row in Case B** — when no catalog match crosses threshold, render "sounds like vintage crooner / 80s synth-pop" via the MuQ-MuLan text branch over a curated vocabulary.
5. **Catalog scale to ~10K then ~50K+** via full MTG-Jamendo + FMA bulk ingest. The FAISS Flat backend is wired and waiting.
6. **Deploy to new Vercel + new HF Space** under the Dundo identity, with PiedPiper's deploys running in parallel until traffic switches over.

## How to read this project at a glance

- **The product**: read `README.md`.
- **The product spec**: read `factory/artifacts/PIVOT_PRD.md`.
- **The engineering decisions**: read the ADRs at `docs/decisions/0001-0005.md`. ADRs 0006+ will land in Dundo.
- **The reviews**: read `factory/artifacts/CODEX_DUNDO_PIVOT_REVIEW.md` + `_RESPONSE.md`, and `factory/artifacts/PERPLEXITY_DUNDO_MARKET_RESEARCH.md`.
- **The historical PiedPiper arc**: read `factory/artifacts/_PREPIVOT/` (45 files) — every Codex review, every phase plan, every implementation note that PiedPiper went through from June 4 to June 19.
- **The why behind it all**: this file.
