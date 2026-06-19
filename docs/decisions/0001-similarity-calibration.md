# ADR-0001: Calibrate similarity scores at the presentation layer; do not display raw cosine as a percent

**Status**: Accepted
**Date**: 2026-06-12
**Decider**: Rajat Arora (with input from Perplexity and Claude-Code reviews)

---

## Context

PiedPiper retrieves the top-K closest tracks in a 160-track reference catalog using L2-normalized cosine similarity over LAION-CLAP music-tuned 512-d audio embeddings. The retrieval ranking is sound — the LOO eval shows R@1=0.394 / R@3=0.494 / MRR=0.458 over the catalog, which is reasonable for a small catalog with the encoder we picked.

The problem we discovered live: when a real Suno generation ("Blacktop Halo-2", a Phonk track) is uploaded, the top-3 displayed matches all render as "100% similar" (originally) or "99.8% / 99.7% / 99.7%" (after the 1-decimal display fix). The raw cosines are 0.998, 0.997, 0.997 — narrowly spread but correctly ordered.

### Empirical evidence (collected 2026-06-12)

Pairwise cosine distribution across all 25,440 catalog-vs-catalog pairs:

| Statistic | Value |
|---|---|
| min | 0.694 |
| max | 0.999 |
| mean | 0.967 |
| median | 0.976 |
| p1 | 0.857 |
| p25 | 0.956 |
| p75 | 0.987 |
| p95 | 0.995 |

**80% of all unrelated-pair cosines sit in [0.95, 1.00].** This is intrinsic to the embedding space, not a bug in our code.

### Root cause: embedding anisotropy

LAION-CLAP is contrastive-trained (InfoNCE loss) and L2-normalizes outputs into a shared latent space. Like every other contrastive encoder (CLIP, BERT, MERT, MuQ-MuLan), it suffers from **embedding anisotropy**: the training objective rewards making positive pairs close but does not force distant pairs to be far from everything, so the entire embedding distribution collapses into a narrow cone in 512-d space. Cosines stay high; useful similarity differences get buried in the third decimal place.

References:
- Dev.to writeup: *Cosine Similarity Lies. Here's What to Use When Your Embeddings All Cluster at 0.85*
- LAION-CLAP HF model card and InfoNCE training description
- Adjacency-based clustering literature (arXiv 1811.02775)

---

## Decision

Keep cosine similarity for **retrieval ranking** (it works correctly). **Do not** show raw `cosine * 100` as the user-facing percentage. Instead, build a presentation-layer calibration pipeline:

### Two-stage scoring

**Stage 1 — Retrieval (unchanged math)**
- Compute windowed CLAP embeddings of the upload.
- Cosine-sweep against the catalog.
- Take top-K candidates ordered by `meanPooledSimilarity`.

**Stage 2 — Presentation calibration**
- Precompute the catalog's pairwise cosine distribution once at startup (25,440 sorted cosines).
- For each retrieved neighbor's raw cosine, look up its **percentile rank** in that distribution. A cosine of 0.998 might be the 99.2nd percentile of "typical similarity in our embedding space"; a cosine of 0.95 might be the 25th percentile.
- Derive a `similarityLabel` from percentile rank:
  - `>= 0.95` → "very close"
  - `>= 0.80` → "close"
  - `>= 0.50` → "moderate"
  - `< 0.50`  → "weak"
- Compute a `querySpecificity` score: fraction of the catalog that scores BELOW a high-similarity threshold (say 0.95) with this query. A query that's broadly similar to most of the catalog (low specificity) gets a "this is a generic generation pattern" annotation in the UI.
- Light segment-level re-ranking: when the top two candidates are within 0.005 on `meanPooledSimilarity` but candidate B has materially higher `maxSegmentSimilarity`, promote B.

### Wire shape additions (locked)

The `/neighbors` response gains these fields per neighbor (existing fields kept for backward compat):

```json
{
  "trackId": "tier2:jamendo:382",
  "rawCosine": 0.998,
  "meanPooledSimilarity": 0.998,
  "maxSegmentSimilarity": 0.997,
  "percentileRank": 0.992,
  "calibratedScore": 0.95,
  "similarityLabel": "very close",
  "segmentSupport": 0.997,
  "track": { ... }
}
```

And a new top-level:
```json
{
  "querySpecificity": 0.31,
  ...
}
```

### UI changes

- Headline shows the `similarityLabel` + percentile, e.g. "**Very close match** · 97th-percentile similarity".
- The raw cosine is shown in small monospace text below: `cosine 0.998 · segment 0.997`.
- Row layout shows the percentile and label, not the raw cosine.
- Specificity < 0.50 triggers a small note: *"This generation pattern is broadly similar to many catalog tracks; the specific match is one of several close candidates."*

---

## Alternatives considered

### A. Swap LAION-CLAP for MERT or MuQ-MuLan

**Rejected.** Anisotropy is a property of contrastive training, not specific to CLAP. MERT and MuQ-MuLan are also contrastive-trained and exhibit the same clustering. Switching costs ~2-3 hours of catalog re-encoding, new model weights download (1-2 GB), and a new Docker image build, in exchange for marginally better discriminative power within an equally narrow cone. Not the right ROI at this stage.

**Reconsider if**: calibration ships and the calibrated UX still feels unconvincing.

### B. Mean-center the embeddings (subtract catalog centroid + L2-renormalize)

**Rejected (in favor of presentation calibration).** Mean-centering is a real, principled fix that does undo the anisotropy at the math layer. But it has costs the presentation approach avoids:

- Shifts the LOO eval numbers (R@1/R@3/MRR all become different numbers — could go up or down; need to re-document).
- Hides the limitation rather than exposing it. The Suno engineer reading the project would see "the cosines are nicely spread" without learning that the underlying embedding space was actually clustered.
- Less honest to the user — the calibration approach exposes both `rawCosine` and `percentileRank`, so a reader can see exactly what the math produces.

**Reconsider if**: the percentile mapping itself proves insufficient and we want a deeper fix.

### C. Simple display rescale (linear map [0.70, 1.00] → [0%, 100%])

**Rejected as primary approach.** Cosmetic only; doesn't surface segment support, percentile context, or specificity. Less informative wire shape for downstream consumers.

### D. Fingerprinting via AcoustID or ACRCloud for exact matches

**Out of scope for this ADR**, but kept in mind as a supplementary signal: ACRCloud is already integrated (Cover Song ID + AI Music Detector). When a query has a high-confidence ACRCloud match, that information is already surfaced in the UI as a separate signal row alongside the calibrated retrieval result. The two signals answer different questions ("is this a cover of X?" vs "what does this sound like?") and should not be merged into a single number.

### E. Build a gold-set of (Suno generation → source-of-inspiration commercial track) pairs and train calibration on that

**Deferred.** This is "Option B" in the existing eval plan (PROJECT_OVERVIEW.md, section 9). Requires generating 30+ Suno tracks targeting seed songs (~$20-40 Suno credits + several hours of curation). Worth doing post-warm-intro if there's appetite for the deeper eval.

---

## Consequences

### Positive

- The displayed score becomes interpretable: "97th-percentile match · very close" tells the user something meaningful even when the underlying cosine is 0.998.
- The retrieval math is unchanged → the existing LOO eval numbers stay valid.
- The API exposes both raw and calibrated values → consumers can choose which they want.
- The engineering story for the warm-intro pitch becomes: *"I noticed the cosines clustered (anisotropy), so I exposed both raw and calibrated scoring in the API and presented a percentile-based label in the UI. Here's the catalog distribution that drives the calibration."* That's a credible engineering decision.

### Negative / costs

- The `similarityLabel` thresholds (0.95 / 0.80 / 0.50 percentile cutoffs) are judgment calls. They should be revisited after the warm-intro window with real user feedback if there is any.
- The `querySpecificity` score is a heuristic. The choice of threshold (0.95 cosine for "this counts as similar") is arbitrary and will need tuning if it produces too many "generic" warnings.
- The `percentileRank` is computed against the catalog distribution, which is small (160 tracks). If the catalog expands to 5K+ tracks, the distribution should be recomputed (which happens automatically at startup).
- More fields in the wire shape means more places for breaking changes. We commit to keeping `meanPooledSimilarity` and `maxSegmentSimilarity` as stable aliases.

### Eval impact

The LOO retrieval eval (R@1, R@3, MRR) is unaffected because retrieval ranking is unchanged. The negatives histogram in `eval.json` continues to reflect raw cosine distribution; a future eval extension may add a "percentile" histogram alongside it.

---

## Implementation tracker

- [x] Backend: precompute catalog distribution at startup (`api.py` lifespan).
- [x] Backend: add percentile lookup, similarity label, specificity score (`similarity.py`).
- [x] Backend: extend `/neighbors` response shape with new fields.
- [x] Frontend: helpers for percentile / label rendering (`lib/api.js`).
- [x] Frontend: update `SimilarityReport` headline to show label + percentile.
- [x] Frontend: update `SimilarityRow` to show percentile and label.
- [x] Tests: update `api.test.js` for the new derived fields.
- [x] Deploy: HF Space restart, Vercel auto-deploy.
- [x] Verify: re-upload Blacktop Halo-2; confirm headline reads "very close · 99th-percentile" instead of "99.8% similar".

(Implemented in commit `ad9c6e4`. Verification: live backend returned the
calibrated wire shape with rawCosine=0.9982 / percentileRank=0.9979 /
similarityLabel="very close" / querySpecificity=0.125 — triggering the
generic-query note.)
