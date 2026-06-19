# ADR-0005: RAG narrative layer + spectrogram visual match

**Status**: Accepted
**Date**: 2026-06-16
**Decider**: Rajat Arora (informed by the Gauntlet-AIDP [rag-cookbook](https://github.com/Gauntlet-AIDP/rag-cookbook) ladder + two rounds of independent Codex plan review)

---

## Context

After [ADR-0004](0004-multi-criterion-similarity.md) decomposed similarity into four interpretable criteria (tempo / key / harmonic / timbre), each match arrives with a defensible per-criterion comparison. Two gaps remained:

1. **No explanatory layer.** A senior reviewer who looks at a top match still has to read a criteria table and synthesize *"what does this mean for the song?"* themselves. The retrieval is the **R** of RAG; the **G** is missing.
2. **No visual evidence.** Users who can't trust their ears (the user's own framing) have no way to see why two pieces of audio resemble each other. Criteria bars summarize; they don't show the time-local signal.

The rag-cookbook organizes RAG into a five-rung ladder:

| Rung | Pattern | Adds |
|---|---|---|
| 1 | **Naive RAG** | Chunk → embed → vector search → top-K |
| 2 | **Metadata Filtering** | Pre-filter docs by attributes before vector search |
| 3 | **Hybrid Search** | BM25 keyword + semantic vector via RRF |
| 4 | **Graph RAG** | Entity/relationship graph traversal |
| 5 | **Agentic RAG** | Agent dynamically picks retrieval strategy |

The cookbook's philosophy is explicit: progressive complexity, refuse to climb without measured evidence, every rung carries its own evaluation gates. It names "premature climbing" as the most common RAG mistake.

PiedPiper sits at **Rung 1 (Naive RAG)** for retrieval — MuQ-MuLan cosine sweep over a 155-track NumPy matrix — plus a strong dose of **metadata-grounded generation** at presentation time: ADR-0004's MIR criteria block + Jamendo enrichment + ACRCloud Cover Song ID are surfaced alongside every match and feed downstream prompts. Codex round-2 review pushed back on calling this "Metadata Filtering" — that term means *pre*-filtering retrieval, which PiedPiper does not do. The accurate term is **metadata-grounded generation**, and this ADR uses it consistently.

---

## Decision

Add a **three-tab interface** inside the existing `SectionComparePanel` row expansion, plus a `POST /narrative` backend endpoint that powers two of the three tabs. Honor the cookbook's "refuse to climb without measured evidence" rule: this is **not** a climb to Hybrid / Graph / Agentic. The LLM narrative is **an explanatory layer over Rung 1 retrieval, not a new retrieval rung.**

### The three tabs

1. **"Why these are similar"** — `POST /narrative` mode `whySimilar`. GPT-4o-mini receives structured metadata about the matched neighbor (criteria values, timestamps, raw cosine, ACRCloud signal) and emits a 80-140 word grounded narrative with structured citations.
2. **"Make mine more distinctive"** — `POST /narrative` mode `creatorAdvice`. Same metadata + same model, different prompt: 60-120 words of concrete creator-feedback tied to the specific criterion that drove the match.
3. **"Visual match"** — no LLM. WaveSurfer.js Spectrogram plugin renders two stacked spectrograms (query + catalog match) with a red overlay band marking the matched 10-second window. Codex round-2 Q6: "show what the bars cannot" — time-local structure, energy bands, vocal/instrument texture.

Tabs lazy-load: clicking a tab fires its data dependency on first click, the result caches across re-clicks within the session.

### Stateless signed `contextToken` (the Codex round-1 Q3 fix)

Round-1 review pushed back on an in-memory cache of `/neighbors` responses: a TTL cache breaks across HF restarts, multiple workers, and page refreshes. The replacement is a stateless HMAC-signed opaque token:

- `/neighbors` returns `contextToken` containing the per-neighbor NarrativeContext fragments (title, artist, queryWindow, matchWindow, rawCosine, criteria), the queryFingerprint (sha256 of upload bytes), the current modelSha + catalogSha, and an `expiresAt` (30-minute TTL).
- `/narrative` accepts `{contextToken, trackId, mode}`. Verifies signature → not expired → modelSha matches current load → catalogSha matches current load → trackId is in the token's allowlist. Rebuilds NarrativeContext from the verified claims.

The client never re-supplies the context; the token IS the context. Anyone who tampered with the token would invalidate the signature; anyone who replayed a stale token would get `412 token-expired` or `412 stale-token` (after a catalog redeploy).

### Canonical SHA-256 cache key (the Codex round-1 Q4 fix)

The in-process LLM cache is keyed by `sha256(canonical_json({...}))` where the dict includes:

- `model_id`, `model_sha`, `catalog_sha`
- `prompt_template_hash` — sha256 of the system+user prompt template strings (so prompt edits auto-invalidate)
- `response_schema_version` (literal `"v1"`)
- `criteria_algorithm_version` (literal `"adr-0004-v1"`)
- `query_fingerprint`
- `track_id`, `mode`
- `criteria_rounded` — sorted by `id`, every numeric value rounded to 3 decimals
- `raw_cosine` — rounded to 3 decimals (the prose can cite cosine, so it must be cache-key-stable)

Plain Python `hash()` is unstable across processes and float reorderings; this canonical form survives both.

### Structured citation validation (the Codex round-2 Q6 fix)

Every LLM response is parsed into a Pydantic `NarrativeResponse` with `citations: list[StructuredCitation]`. Each citation declares `trackId`, `side` (`"query"` or `"match"`), `timestampRange`, `criterionIds`, and `citedValues` whose keys must take the form `"<criterionId>.queryValue"` / `"<criterionId>.matchValue"` / `"rawCosine"`. The backend validates every citation against the supplied context:

- `trackId == context.trackId` (the LLM never cites a different track).
- Every `criterionIds[i]` exists in `context.criteria`.
- Numeric `citedValues` match the corresponding context value within tolerance (±2 BPM for tempo, ±0.01 for cosine, exact match for strings).
- `timestampRange` lies within the `side`-appropriate window ± 0.5 s.

Any failure → `NarrativeUnavailable(reason="citation-hallucinated")`. **No retry.** The hallucinated response is discarded rather than rendered.

### Context-completeness gate (the Codex round-2 Q2 fix)

Before any OpenAI call, return `LowConfidence` if context completeness fails:

- `criteria` is empty or null → `reason="missing-criteria"`.
- `title` is missing → `reason="missing-metadata"`.
- Both windows must be well-formed (start < end, non-negative).
- At least one criterion has `agreement >= 0.55` **OR** `rawCosine >= 0.75` → otherwise `reason="weak-evidence"`.

This gate is cookbook-named: it borrows Agentic RAG's "self-evaluation gate" downward as a cost guardrail. The frontend renders typed copy per `reason` rather than showing a generic error.

### The LLM does not hear audio

Embedded as a non-negotiable system-prompt rule in every mode:

> You receive structured metadata about two audio segments. You do not hear the audio. You do not determine copyright infringement, ownership, or legal status. Cite only tracks, criteria, and values present in the supplied context. Output a single JSON object matching the schema.

This rule is in the prompt, locked in this ADR, and surfaced in the README. It is the system's load-bearing trust contract: the LLM is a **post-hoc interpreter of structured retrieval results**, not a music analyzer.

### Cost framing

Honest framing, never a guarantee:

- Per-request cost is **single-digit cents to low dollars** depending on prompt size, cache hit rate, and OpenAI pricing.
- Cost guardrails: lazy load (LLM only fires on tab click), canonical SHA-256 cache (re-clicks free), prompt cap (8 KB), `max_tokens` cap (400), no retry loops, no-key disable path (503 narrative-disabled).
- Cache misses after redeploy: every prompt-template edit, model change, or catalog regen invalidates cache automatically via the included hashes. Expected.

The README does not quote a fixed per-request cost.

---

## What this ADR deliberately does NOT do

- **Does not climb the cookbook ladder.** We are still at Rung 1 retrieval with metadata-grounded generation at presentation time. Hybrid (no text query), Graph (no measured evidence), Agentic (premature) are all out. Any future climb requires an eval showing the current rung is the bottleneck.
- **Does not introduce a vector DB.** NumPy matmul over 155 tracks is sub-millisecond; a vector DB would be architecture theater. Revisit at ~10k+ tracks.
- **Does not pass audio bytes to the LLM.** Structured metadata only. Locked in the system prompt + this ADR + the README.
- **Does not accept arbitrary client-supplied prompt context.** All context flows through the HMAC-signed token; the client cannot inflate prompts.
- **Does not retry LLM failures.** One call, one result. Cost-bounded.
- **Does not claim a fixed per-request cost.**
- **Does not determine copyright.** Acoustic-similarity language only, as established in earlier ADRs.

---

## Wire-shape contract

### `POST /neighbors` (additions)

```json
{
  ...existing fields...,
  "queryFingerprint": "<sha256 hex of upload bytes>",
  "contextToken": "<base64url(json).<hmac_sha256_hex>>"
}
```

When `CONTEXT_TOKEN_HMAC_KEY` is unset, `contextToken: null` and `/narrative` returns 503.

### `POST /narrative`

Request:

```json
{
  "contextToken": "<from /neighbors>",
  "trackId": "tier1:itunes:380907765",
  "mode": "whySimilar" | "creatorAdvice"
}
```

Success response (`200`):

```json
{
  "kind": "narrative",
  "mode": "whySimilar",
  "prose": "...",
  "citations": [
    {
      "trackId": "tier1:itunes:380907765",
      "side": "query",
      "timestampRange": [20.0, 30.0],
      "criterionIds": ["tempo", "key"],
      "citedValues": {
        "tempo.queryValue": 100.0,
        "tempo.matchValue": 100.5,
        "key.matchValue": "C major",
        "rawCosine": 0.881
      }
    }
  ]
}
```

Typed soft-failures (`200`):

```json
{"kind": "low_confidence", "reason": "missing-criteria" | "missing-metadata" | "weak-evidence" | "context-cap-exceeded"}
{"kind": "unavailable",   "reason": "malformed-llm-output" | "openai-error" | "citation-hallucinated"}
```

Typed hard-failures (`4xx` / `5xx`):

| Status | `error` code | Cause |
|---|---|---|
| 503 | `narrative-disabled` | OPENAI_API_KEY or CONTEXT_TOKEN_HMAC_KEY missing, or rag_narrative module unimportable |
| 401 | `invalid-token` | HMAC signature mismatch |
| 400 | `malformed-token` | Bad token shape |
| 412 | `token-expired` | Past `expiresAt` |
| 412 | `stale-token` | modelSha or catalogSha changed since issuance |
| 404 | `not-in-context` | trackId wasn't in the token's allowlist |
| 422 | `unsupported-mode` | mode not in {whySimilar, creatorAdvice} |
| 422 | `malformed-context` | Token fragment failed to materialize into NarrativeContext |
| 500 | `narrative-error` | Unexpected backend exception |

---

## Observability

The /narrative layer ships with in-process telemetry — the right rung for this scale (a Prometheus/Datadog stack would be overbuilt at demo traffic):

- **`backend/backend/narrative_telemetry.py`** — counters (`total_calls`, `by_mode`, `by_kind`, `by_error`, `openai_calls`, `gate_short_circuits`), latency sliding window (256 samples, p50/p95/p99), cost estimate in cents (char-based × GPT-4o-mini pricing — directional, not accounting-grade), structured logger (one INFO line per call with `mode=... kind=... latency_ms=... openai_called=...` fields), Sentry tag helper.
- **Unknown kind / error values bucket to `"_other"`** sentinel rather than growing arbitrary counter keys per typo. Operators see `_other > 0` and know to update the known-value sets.
- **`GET /narrative/stats`** — JSON snapshot for live visibility into what the layer is doing right now. Counters reset on restart; this is a "right now" snapshot, not a long-term metrics store.
- **Frontend Sentry breadcrumbs** — `fetchNarrative` drops a breadcrumb on every outcome (success kind, typed error code, network failure) into the app's existing Sentry instance. Tag-style fields so failures aggregate by `mode` + `kind` + `code` in the existing dashboard.

## RAG eval harness

Per the rag-cookbook's "every rung has its own evaluation gates" principle:

- **Golden set** at `backend/tests/fixtures/narrative_golden_set.json` — 12 hand-crafted cases across 5 categories: happy path (3), low context (3), hallucinated citation (4), malformed output (1), OpenAI error (1).
- **Harness** at `backend/backend/scripts/run_rag_eval.py` — runs each case through `generate_narrative()` with `_call_openai_json` patched to return the golden-mapped response, scores aggregate metrics (kind agreement, reason agreement, gate-respected rate), writes a summary to `factory/artifacts/RAG_EVAL_RESULT.json`.
- **Five baseline gates** that must hold at 1.0:
  - `happy_path_kind_agreement` — valid LLM output must return `kind=narrative`.
  - `low_context_gate_correctness` — low-context cases must short-circuit before the LLM call.
  - `hallucination_rejection` — citation hallucinations must be caught and surfaced as `unavailable`.
  - `malformed_rejection` — schema-invalid LLM output must surface as `unavailable, reason=malformed-llm-output`.
  - `openai_error_handling` — SDK-helper returning None must surface as `unavailable, reason=openai-error`.
- **Pytest gate** at `backend/tests/test_rag_eval.py` — runs the harness in-process on every CI build and fails the suite on any baseline regression. Zero API spend (offline only).

The harness is fully offline. Adding real-OpenAI evals against a separate manual golden set is a future direction (gated behind a cost / `RUN_LIVE_EVAL=1` env flag) but is not load-bearing for this ADR.

## Verification

- **58 narrative-layer tests** across seven files (177 backend tests total, up from 119 pre-Commit A):
  - `backend/tests/test_rag_narrative.py` (8 fixtures, all OpenAI-mocked) — valid path, malformed LLM JSON, OpenAI helper returning None, hallucinated criterion, wrong trackId, low-context short-circuit, cache key order stability, cache key changes when prompt template changes.
  - `backend/tests/test_context_token.py` (10 fixtures) — valid roundtrip, tampered signature, tampered payload, expired token, stale modelSha, stale catalogSha, malformed string, missing HMAC key, env reflection, fragment shape.
  - `backend/tests/test_narrative_endpoint.py` (15 fixtures) — 503 no-OpenAI-key, 503 no-HMAC-key, 422 unsupported-mode, 401 invalid signature, 400 malformed token, 412 expired, 412 stale model, 412 stale catalog, 404 not-in-context, 200 happy path, 200 LowConfidence, 200 NarrativeUnavailable, `/narrative/stats` snapshot, `/narrative/stats` empty, `/narrative/stats` tracks errors.
  - `backend/tests/test_narrative_telemetry.py` (13 fixtures) — counter increments, gate counter, error counter, unknown-value bucketing, latency percentiles, window cap, cost estimate scaling, context manager normal + exception paths, structured log line shape, reset.
  - `backend/tests/test_rag_eval.py` (10 fixtures) — golden set loads, harness runs clean, five baseline gates each at 1.0, low-context cases never call the LLM, eval output writes to the expected path.
  - `backend/tests/test_narrative_e2e.py` (2 fixtures) — `/neighbors → contextToken → /narrative` wire-shape contract roundtrip + rebuilt context passes through `generate_narrative()`.
- **No regressions** in the existing 119 backend tests after Commit A merged.
- **Frontend**: 20/20 vitest tests pass; `vite build` clean; WaveSurfer + Spectrogram chunks (12.5 KB + 13.9 KB gz) are lazy.

---

## Consequences

- **Senior reviewer experience.** The match expansion now answers "similar how?" with grounded prose, not just a criteria table. The visual tab gives users who can't trust their ears a spectrogram view of the matched window.
- **Cookbook discipline.** The README and this ADR both name PiedPiper's rung position honestly: Rung 1 retrieval + metadata-grounded generation at presentation time, NOT a climb. Future PRs that propose climbing must come with an eval first.
- **Stateless backend.** No in-memory cache, no session affinity, no Redis dependency. Survives HF Space restarts and worker rotations.
- **No frontend dependency on live OpenAI behavior.** When `OPENAI_API_KEY` is unset (dev environments, CI), `/narrative` returns 503 and the frontend tabs show the no-key fallback panel. The headline retrieval flow is unaffected.
- **No prompt-tampering vector.** The signed `contextToken` is the only path by which prompt context reaches the LLM; the client cannot inflate prompt size or substitute different track metadata.

---

## Open questions

- **Latency of the OpenAI call.** First-click target is < 5 s. If cold-call latency at deploy regularly exceeds that, consider a streaming SSE response variant. Out of scope for this ADR.
- **Cache size cap.** The in-process LRU is currently unbounded. At demo traffic (~hundreds of unique matches), memory is trivial; productionizing would add a max-entries cap.
- **Audio-aware future evolution.** A natural follow-up would feed an audio CLIP-style encoder a (query, candidate) pair to ground the narrative in the audio itself — a *true* multi-modal layer. Out of scope at this rung; the metadata-grounded approach earns its keep at v1 demo scale.

---

## References

- ADR-0001 — calibrated similarity display
- ADR-0002 — LAION-CLAP → MuQ-MuLan encoder swap
- ADR-0003 — density-relative calibration at catalog scale
- ADR-0004 — multi-criterion similarity layer (the metadata this ADR grounds the LLM in)
- [rag-cookbook](https://github.com/Gauntlet-AIDP/rag-cookbook) — the rung ladder this ADR positions PiedPiper on
- `factory/artifacts/CODEX_RAG_NARRATIVE_PLAN_REVIEW.md` + `CODEX_RAG_NARRATIVE_PLAN_REVIEW_RESPONSE.md` — round-1 review
- `factory/artifacts/CODEX_RAG_REVISED_PLAN_REVIEW.md` + `CODEX_RAG_REVISED_PLAN_REVIEW_RESPONSE.md` — round-2 review
- `factory/artifacts/CODEX_RAG_NARRATIVE_IMPLEMENTATION.md` + `CODEX_RAG_NARRATIVE_IMPLEMENTATION_FEEDBACK.md` + `CODEX_RAG_NARRATIVE_IMPL_NOTES.md` — implementation hand-off
- `factory/artifacts/HF_SPACE_SECRETS_FOR_COMMIT_C.md` — deploy secrets checklist
