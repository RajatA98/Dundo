# ADR-0004: Decompose similarity into four interpretable criteria + time-anchored section comparison

**Status**: Accepted
**Date**: 2026-06-15
**Decider**: Rajat Arora (informed by the deep research at `~/Documents/Music_Similarity_Criteria_Research_20260615/Music_Similarity_Criteria.md`)

---

## Context

PiedPiper currently returns top-K nearest neighbors from a MuQ-MuLan cosine sweep (ADR-0002) with calibrated percentile + similarity-label display on top (ADR-0001). The live demo surfaced two user-side problems:

1. **One cosine doesn't defend a match.** When a Suno Phonk generation matches "Blinding Lights — The Weeknd" at cosine 0.881, the report shows that single number plus a calibrated label. A senior reviewer asks "similar *how*?" and the system has no answer. A creator asks "what should I change?" and the system has no actionable feedback.
2. **The percentile display "seems inaccurate"** to the user (their words). Even after ADR-0001's calibration work, a percentile-rank readout in the catalog's own distribution is interpretable to engineers, not creators. They explicitly asked for it to be removed.

A research sweep ([report on disk](../../../../../Documents/Music_Similarity_Criteria_Research_20260615/Music_Similarity_Criteria.md)) found that the MIR field, and the four named commercial tools (Cyanite, MIPPIA, Soundverse Trace, ACRCloud), have converged on a small interpretable axis set — typically melody, harmony, rhythm, with some tools adding timbre or instrumentation. The same research established that the four convergent axes are all cheap to compute with librosa, an existing PiedPiper dependency, in ~350 ms of CPU per upload.

The honest framing of this ADR: we are not inventing a similarity taxonomy. We are picking the smallest defensible subset of an industry-standard one, computing it ourselves because Spotify deprecated their audio_features endpoint in November 2024 with no replacement, and surfacing it in the time-anchored shape the four-tool audit shows is the consensus UX.

---

## Decision

Lock **four criteria** as PiedPiper's multi-criterion similarity layer for v1:

### The four locked criteria

| # | Criterion | librosa method | What it measures | User-facing label pattern |
|---|---|---|---|---|
| 1 | **Tempo** | `librosa.beat.beat_track` | BPM | "128 BPM ↔ 132 BPM — 4 BPM apart" / "same tempo" |
| 2 | **Key + mode** | chroma + Krumhansl-Schmuckler | tonal center | "A minor ↔ A minor — same key" / "C major ↔ A minor — relative" |
| 3 | **Harmonic content** | `chroma_cens` mean + cosine | which pitch classes are present | "similar chord palette" / "different chord palette" |
| 4 | **Timbre** | `mfcc` mean+std + cosine | production / instrument feel | "similar production feel" / "different production" |

These cover the **three consensus axes** named explicitly by MIPPIA + Soundverse Trace (melody = chroma; rhythm = tempo; harmony = key/mode), plus **timbre** — Cyanite's instrumentation axis, which is the dimension a Suno engineer will recognize most directly because production aesthetic is the primary output of Suno's models.

### The wire-shape contract

`/neighbors` extends the per-neighbor object with a `criteria` block:

```json
{
  "trackId": "tier1:itunes:1488408568",
  "rawCosine": 0.881,
  "meanPooledSimilarity": 0.881,
  "maxSegmentSimilarity": 0.895,
  "percentileRank": 0.970,
  "similarityLabel": "very close",
  "matchTimestamp": { ... },
  "track": { ... },
  "criteria": {
    "tempo":    { "queryValue": 128, "matchValue": 132, "agreement": 0.85, "label": "4 BPM apart" },
    "key":      { "queryValue": "A minor", "matchValue": "A minor", "agreement": 1.00, "label": "same key" },
    "harmonic": { "agreement": 0.84, "label": "similar chord palette" },
    "timbre":   { "agreement": 0.71, "label": "similar production feel" }
  }
}
```

Every criterion entry carries the same shape: a numeric `agreement` in [0, 1] and a categorical `label` for the headline display. Tempo and key additionally carry `queryValue` + `matchValue` because those are interpretable on their own; harmonic and timbre do not, because the underlying numerics (12-d chroma cosine, 26-d MFCC cosine) don't surface a meaningful raw value to a user.

`percentileRank` and `similarityLabel` are kept in the wire shape for backward compatibility but **deprecated as headline displays** — see the UI contract below.

### The per-criterion thresholds (locked here, not adjustable per deploy)

These are judgment calls and need to be locked to make the labels stable. Adjusting them later requires an ADR amendment.

**Tempo**:
- `|q_bpm - m_bpm| <= 3` → "same tempo" (agreement 1.0)
- `|q_bpm - m_bpm| <= 10` → "{Δ} BPM apart" (agreement 1 - Δ/20)
- otherwise → "{Δ} BPM apart" (agreement max(0, 1 - Δ/40))

**Key + mode**:
- exact key + mode match → "same key" (agreement 1.0)
- relative major/minor (A minor ↔ C major) → "relative" (agreement 0.7)
- perfect fifth apart → "fifth apart" (agreement 0.5)
- otherwise → "different key" (agreement 0.0)

**Harmonic (chroma cosine)**:
- `>= 0.85` → "very similar chord palette" (agreement = cosine)
- `>= 0.65` → "similar chord palette"
- `>= 0.40` → "moderate chord overlap"
- otherwise → "different chord palette"

**Timbre (MFCC cosine on a 26-d mean+std vector)**:
- `>= 0.80` → "very similar production feel"
- `>= 0.55` → "similar production feel"
- `>= 0.25` → "moderately different production"
- otherwise → "different production"

The agreement values feed the bar widths in the criteria UI; the labels feed the headline display per row.

### The UI contract

**`SimilarityRow`** (the per-row table):
- Replace the percentile column at the right edge with a **single-line similarity label** (per the user's deferred decision in the previous round): just "Very close" / "Close" / "Moderate" / "Weak" derived from `similarityLabel`, with the raw cosine ("cos 0.881") in small monospace beneath it. **No percentile number anywhere.**
- Add a disclosure chevron at the row's left edge. Click to expand the new `SectionComparePanel`. Opening one row auto-closes others.

**`SimilarityReport`** (the headline block):
- Replace "Very close · 97th percentile match" with **"Very close · cosine 0.881"**. The calibrated label is kept; the percentile readout is dropped. Raw cosine plus segment cosine show in the existing small-text caption.

**`SectionComparePanel`** (new — the expanded panel under each row):
- Header strip: existing `matchTimestamp` rendered as "match: query 0:40–0:50 ↔ track 0:10–0:20".
- Two side-by-side **windowed snippet players**: query snippet (uses the uploaded `File` via `URL.createObjectURL`, anchored to `queryStartSec`–`queryEndSec`) + match snippet (uses the catalog track's audio URL, anchored to `catalogStartSec`–`catalogEndSec`). Both use the new `AudioPlayer` windowed mode (extended with `startSec` / `endSec` props that append `#t=` to the src URL and clamp playback to the window).
- Two side-by-side **spectrograms** of the same windows. Rendered client-side by WaveSurfer.js's spectrogram plugin (~50 KB bundle add). No backend round-trip.
- **The four-criterion table**: one row per locked criterion. Each row shows the label, the bar-shaped agreement score, and the queryValue/matchValue when available.

Backward compatibility: the panel renders sensibly even when `criteria` is missing from the response (e.g., during the brief deploy window between backend and frontend ships). Missing criteria → panel shows just the snippet players + spectrograms.

---

## Alternatives considered

### A. Reuse Spotify's `audio_features` endpoint to get tempo, key, energy, valence, danceability for catalog tracks

**Rejected.** Spotify deprecated `audio_features` on 2024-11-27 with no replacement; new apps get 403s. Their May 2025 quota change raised the extended-API floor to 250K monthly active users, effectively unreachable for a portfolio project. Even if the endpoint worked, it would only cover tracks in Spotify's catalog, not the user's Suno upload — so a librosa-side computation would still be required for the query. Implementing both would double the surface; implementing only librosa covers both sides consistently.

**Reconsider if**: Spotify reopens the endpoint (no signal that's coming).

### B. Train a custom classifier for mood / valence / arousal / danceability (Cyanite-style)

**Rejected for v1.** Cyanite's mood vector is one of their strongest displays, but it requires either a trained classifier (heavy: training infrastructure, labeled data, model serving) or a heuristic mapping from acoustic features (low quality: research-grade, not product-grade). The librosa-native shortcuts produce noisy mood estimates that wouldn't survive a senior reviewer's scrutiny.

**Reconsider if**: the project pivots to commercial deployment and mood classification becomes load-bearing. At that point, fine-tune a model on a labeled dataset.

### C. Surface MIPPIA-style bar-level (4-bar) alignment instead of fixed 10-second windows

**Rejected for v1.** Bar-level alignment is musicologically tighter than fixed 10-second windows — but requires reliable downbeat tracking, which librosa can do but inconsistently for polyrhythmic, modal, or electronic music. The 10-second windows PiedPiper already uses (locked from PROJECT_PLAN Phase 2) trade musicological precision for engineering simplicity, and the trade is the right one for v1.

**Reconsider if**: downbeat tracking matures (e.g., a tencent-ailab beat-tracking model lands on HuggingFace with the same kind of step-change MuQ-MuLan represented for embeddings).

### D. Add time signature, onset density, spectral contrast as additional criteria

**Tier A in the research, deferred.** Each adds modest value at modest cost. They can land in a follow-up commit once the four Tier S criteria are shipping cleanly; adding them now risks UI clutter (four criteria fit cleanly; seven start to feel like a Spotify audio-features clone).

**Reconsider if**: user testing shows the four locked criteria aren't enough to defend "similar."

### E. Heavyweight XAI — SHAP per-feature contribution, attention map overlays

**Out of scope.** The research's Finding 4 documents these as real research directions but heavier engineering than the project warrants. Explicit feature decomposition (the MIPPIA / Soundverse pattern) is the light-touch XAI that lands the demo without the engineering tax.

### F. Backend-rendered spectrograms in the `/neighbors` response

**Considered, rejected in favor of client-side.** Backend rendering avoids the WaveSurfer.js bundle add (~50 KB) and produces consistent visuals across browsers. But it adds 5-15 KB per match to the response (small PNG per spectrogram), adds backend latency (matplotlib rendering ~100 ms per spectrogram), and complicates the spectrogram-anchoring logic (the backend would need to know exactly which window the frontend wants). Client-side keeps the backend clean and lets the spectrogram render lazily when the user opens the disclosure.

---

## Consequences

### Positive

- **The display answers "similar how?" directly.** Four named axes, each with a queryValue / matchValue / agreement / label, plus the snippet audio + spectrogram for visual evidence. No senior reviewer should be able to ask "but what does cosine 0.881 *mean*?" after seeing this.
- **The Spotify-vocabulary advantage.** A Suno engineer parses "Tempo: 128 BPM" and "Key: A minor" instantly because those are the Spotify field names they've worked with for a decade. PiedPiper picks up the vocabulary the industry has built around without depending on Spotify.
- **No new heavy dependencies.** librosa is already in the dep tree. The only new frontend dep is WaveSurfer.js for spectrograms. Backend stays Python-pure.
- **Time-anchored evidence is now visible**, not just numerical. The user's "audio overlay" ask is addressed by the side-by-side snippet players + side-by-side spectrograms.

### Negative / costs

- **Per-criterion threshold choices are judgment calls.** The locked thresholds in this ADR (3 BPM for "same tempo," 0.85 chroma cosine for "very similar chord palette," etc.) are reasonable defaults but not validated against a labeled benchmark. Future ADR amendment is the right path if user testing surfaces problems.
- **librosa key detection is ~70-80% accurate on Western tonal music** and noticeably worse on atonal, modal, polyrhythmic, or electronic music. The "same key" label will sometimes be wrong, especially on Suno-generated Phonk or hip-hop. The UI should not overclaim certainty (the label "same key" reads as definitive; "approximately A minor" reads as appropriately hedged — defer that copy choice to implementation).
- **Catalog backfill is required.** The existing 155-track corpus has no `mir_features` field. A new script (`backend/backend/scripts/enrich_mir_features.py`) walks each entry, runs the feature stack, and writes the values back. Same idempotent pattern as `enrich_jamendo.py`. No MuQ re-encode required; the new features are independent of the embedding.
- **The four criteria are interpretable, not exhaustive.** Real perceptual music similarity has dozens of dimensions (vocal characteristics, lyrical content, era, mix style, etc.). The four chosen axes are a defensible minimum. ADR-0004's claim is that they cover the consensus, not that they replace deeper musicological analysis.
- **`SimilarityRow.test` and `deriveHeadline` tests need updating** — both depended on the percentile field as a display contract. The migration is straightforward (replace `percentileRank` assertions with `similarityLabel`) but is a multi-file change.

### Eval impact

The criteria layer is **additive only** — top-K retrieval ordering still comes from `meanPooledSimilarity`. The LOO eval (ADR-0002 §"Eval impact": R@1=0.639, R@3=0.735, MRR=0.692) is unchanged. The verification harness (`backend/backend/scripts/verify_matching.py`, ADR-0002 §"Verification") should continue to return 10/10 self-retrieval.

A follow-up could add a small criteria-side eval: for each self-retrieval test, check that the criteria block shows 1.00 agreement on every axis (since the query IS the catalog track). That's a free regression test and should be folded into the verification harness.

---

## Implementation tracker

- [x] ADR drafted + accepted.
- [ ] `backend/backend/mir_features.py` — `compute(audio, sr) -> MirFeatures` dataclass + four feature extractors.
- [ ] `backend/backend/similarity.py` — `compare_tempos`, `compare_keys`, `compare_chroma`, `compare_timbre` helpers, each returning `{agreement, label}` per the locked thresholds above.
- [ ] `backend/backend/api.py` — at `/neighbors` time, compute query MIR features once, attach `criteria` block per neighbor.
- [ ] `backend/backend/scripts/rebuild_corpus.py` — extend ingest to persist `mir_features` per entry.
- [ ] `backend/backend/scripts/enrich_mir_features.py` — new CLI for backfilling the 155-track catalog without re-encoding.
- [ ] `backend/tests/test_mir_features.py` — unit tests on synthetic fixtures (a sine wave at 440 Hz should detect as key A, a tone-burst at 120 BPM should detect tempo 120 ± 3, etc.).
- [ ] `quality-scorer/src/components/AudioPlayer.jsx` — extend with `startSec` / `endSec` props (windowed playback via `#t=` media fragment).
- [ ] `quality-scorer/src/components/SectionComparePanel.jsx` — new panel with snippet players, spectrograms (WaveSurfer.js), and criteria table.
- [ ] `quality-scorer/src/components/SimilarityRow.jsx` — disclosure chevron + percentile-column drop + similarity-label-only right edge.
- [ ] `quality-scorer/src/components/SimilarityReport.jsx` — headline rewrite ("Very close · cosine 0.881"), drop "Nth percentile match."
- [ ] `quality-scorer/src/lib/api.js` — `deriveHeadline` updated, `criteria`-aware fall-through for old backends.
- [ ] `quality-scorer/src/pages/ScorerPage.jsx` + `ReportCard.jsx` — file-object plumbing so `SectionComparePanel` can play windowed slices of the user's upload.
- [ ] Frontend tests — `api.test.js` headline contract, `SimilarityReport.test` headline rendering.
- [ ] HF Space deploy + verification harness re-run (should stay at 10/10 self-retrieval; criteria block should show all-1.00 agreement on self-matches).
- [ ] README "Key engineering decision" section gets one line linking to ADR-0004.

(Implementation lands in commits referenced by `git log --grep=ADR-0004`. As each box gets checked, the corresponding commit hash goes here.)

---

## References

- [ADR-0001](0001-similarity-calibration.md) — calibration mechanics (percentile + label + querySpecificity). This ADR drops the percentile from user-facing display but keeps the calibrated label.
- [ADR-0002](0002-swap-clap-for-muq-mulan.md) — the encoder swap that the criteria layer sits on top of. Top-K ordering still comes from MuQ-MuLan cosine.
- [ADR-0003](0003-catalog-scale-calibration.md) — scale-survivability of the calibration layer. The criteria layer is also catalog-scale-friendly: features are per-track, comparison is per-pair, no global distribution dependency.
- Research report on disk: `~/Documents/Music_Similarity_Criteria_Research_20260615/Music_Similarity_Criteria.md`.
- librosa docs: https://librosa.org/doc/latest/
- MIPPIA: https://mippia.com/en
- Cyanite: https://cyanite.ai/
- ACRCloud Cover Song ID: https://docs.acrcloud.com/tutorials/recognize-music
- Spotify audio_features deprecation: https://freqblog.com/blog/spotify-audio-features-replacement-2026/
