# ADR-0003: Density-relative calibration is what survives catalog scale

**Status**: Accepted
**Date**: 2026-06-14
**Decider**: Rajat Arora (after an external reviewer brief flagged this as a senior-reviewer gap)

---

## Context

PiedPiper's deployed catalog is ~155 tracks. A senior engineering reviewer reading this project will almost certainly raise the scaling objection:

> *"At a real catalog scale — say a music platform's licensed library, ~10⁷–10⁸ tracks — nearest-neighbor density explodes. Every query will have many tracks at high cosine similarity. The top-1 cosine distribution shifts right as density increases. A fixed similarity threshold and a 'top-3 closest' display will produce high false-positive rates. The 160-track demo doesn't and can't address this."*

The objection is correct. The repo's existing limitations paragraph names "catalog incompleteness" but does not name the harder structural objection (density-driven false-positive blow-up) or describe the calibration mechanism we believe survives it. This ADR closes that gap.

The factual basis for the objection:

1. As N (catalog size) grows, the expected nearest-neighbor distance shrinks toward zero in any fixed-dimensional embedding space ("crowding"). For neural music encoders, which already exhibit anisotropy (ADR-0002 documented LAION-CLAP's mean random-pair cosine at 0.967), the crowding compounds the existing distribution-collapse problem.
2. Any absolute cosine threshold — including PiedPiper's provisional `SIMILARITY_THRESHOLD_DEFAULT = 0.70` — that worked on a 160-track catalog will saturate as N grows. At 10⁷ tracks, every query will exceed the threshold against thousands of catalog tracks.
3. A "top-3 closest" report card without distribution context is structurally unable to discriminate "this query is distinctively close to track X" from "this query is broadly close to everything in the catalog."

Without addressing this, the project reads as a 160-track demo whose architecture doesn't scale — which is the wrong read, because the calibration machinery PiedPiper ships *is* the scaling answer. This ADR makes that explicit.

---

## Decision

**Verdict must come from where a query sits in the catalog's own similarity distribution, not from an absolute cosine.** Two mechanisms together — both already shipped in PiedPiper's calibration layer — do this work and are the answer to the scaling objection.

### Mechanism 1 — Percentile rank (density-relative score)

Documented in [ADR-0001](0001-similarity-calibration.md). At startup, the backend precomputes the catalog's pairwise cosine distribution and sorts it. At query time, each retrieved neighbor's raw cosine is mapped to its percentile rank in that distribution. A cosine of 0.881 might be the 97th percentile at the 160-track scale or the 84th percentile at a 10⁷-track scale; either way, the percentile is a *density-relative* quantity that adapts as the catalog grows. The displayed `similarityLabel` ("very close" / "close" / "moderate" / "weak") is keyed off this percentile, not the raw cosine.

This is the mechanism that lets the verdict remain interpretable as catalog density changes. A "very close" label at 100K tracks means the same thing relative to its catalog as it does at 160 tracks: top 5% of typical music-vs-music similarity.

### Mechanism 2 — Specificity score (false-positive suppression)

Documented in [ADR-0001](0001-similarity-calibration.md) §"querySpecificity." At query time, the backend computes the fraction of catalog tracks whose cosine with the query is below a high-similarity threshold. **High specificity (near 1.0) means the query is distinctively close to a small handful of catalog tracks; low specificity (near 0.0) means the query is broadly similar to most of the catalog.** The deployed UI surfaces a "this generation is broadly similar to many catalog tracks; the specific match is one of several close candidates" note when specificity falls below 0.50.

**At scale, specificity is the precision-preserving mechanism.** A query that hits 80% of a 10⁷-track catalog above 0.95 cosine is a generic acoustic pattern, not a meaningful match. Suppressing or qualifying the headline match in that case is what keeps false-positive rates manageable as N grows. Without specificity, the system reports a top-3 in every case and lets the viewer assume it means something — which is the false-positive failure mode the scaling objection points at.

### Why this answers the objection

Together, percentile + specificity replace the brittle "fixed cosine threshold + top-3 raw list" architecture with a density-aware one:

- **Percentile** rescales the displayed strength of a match to the catalog's own distribution. It does not break as N changes.
- **Specificity** detects when the match isn't really a match — when the query is generic relative to the catalog — and qualifies the headline accordingly.
- The raw cosine and segment-level support are still surfaced in small monospace text for the technically-literate reader who wants the underlying numbers.

The fixed `SIMILARITY_THRESHOLD_DEFAULT = 0.70` remains in the codebase as a transitional artifact. The calibrated label + specificity are the primary verdict-producing mechanisms; the threshold is only the "Completely unique" empty-state cutoff. As the catalog grows, the threshold's role should shrink to zero (the percentile + specificity carry the verdict on their own); this is queued as a follow-up.

---

## What this ADR does NOT claim

This is the boundary the brief explicitly asked be drawn: **the mechanism above is argued, not proven at scale.** The deployed PiedPiper catalog is ~155 tracks, which is too small for the argument to be a measurement. Validating at 10⁵–10⁷ scale requires a larger labeled benchmark and is queued as future work.

Concretely, the open edges:

- **Percentile distributions are themselves catalog-dependent.** Two catalogs of the same size with different genre coverage will produce different percentile mappings for the same raw cosine. This is a feature (the score is meaningful relative to *this* catalog), but it means scores are not directly comparable across deployments without reporting the underlying distribution.
- **Specificity thresholds (the 0.95 / 0.50 cutoffs) are judgment calls.** They should be revisited empirically as the catalog grows or as user feedback accumulates.
- **The specificity score's threshold parameter** (the cosine above which a catalog track "counts as similar to the query") will need to be density-aware too — what counts as "similar" at the 99.5th percentile of a 10⁷-track distribution is not the same as at 160-track scale.
- **The MuQ-MuLan encoder itself has not been measured at catalog-scale workloads.** Its anisotropy improvement over LAION-CLAP (ADR-0002, +62% R@1, 12× discrimination ratio) was measured at 155 tracks; whether the gain compounds at 10⁷ scale or saturates is an empirical question we have not run.

---

## Consequences

### Positive

- The architecture answer to the false-positive-at-scale objection is now documented, linkable, and traceable from both the README and the `/evaluation` page.
- Density-relative calibration is the architecturally correct response. Defending it stays honest because it doesn't claim more than it can measure.
- The specificity score, which currently does relatively quiet work in the UI ("note: this generation pattern is broadly similar to many catalog tracks"), is now framed as the precision-preserving mechanism at scale — which is the role it actually plays. The framing was understating its load-bearing function.

### Negative / costs

- This ADR commits to the "percentile + specificity" architecture as PiedPiper's scaling story. If empirical measurement at scale ever shows it breaks (e.g., the percentile mapping becomes uninformative because all music collapses to the high end of a 10⁷-track distribution), this ADR needs an honest update rather than a silent rollback.
- The "argued, not proven" caveat is now load-bearing for the project's intellectual honesty. It must remain in the public-facing framing as long as the larger eval is not run.

### Open follow-up

- A labeled retrieval eval against Da-TACOS or SHS100K cover-song pairs would be the cheapest credible validation. Both datasets distribute pre-extracted features rather than raw audio (specifically to sidestep the same copyright constraint Tier-1 navigates), making them tractable to run without re-licensing audio.
- A scaling-factor empirical study — sampling the catalog at N=160 / N=1,000 / N=10,000 from MTG-Jamendo and re-measuring the percentile + specificity distributions — would let the "argued at scale" claim become "measured to N=10,000, projected past that."

---

## References

- [ADR-0001](0001-similarity-calibration.md) — calibration mechanics (percentile, label, querySpecificity).
- [ADR-0002](0002-swap-clap-for-muq-mulan.md) — encoder swap that the calibration sits on top of.
- README §"Rights and catalog" — names the demo-vs-production catalog trade-off this ADR addresses on the scaling axis.
- Da-TACOS cover-song dataset (ISMIR 2019): https://mtg.github.io/da-tacos/
- SHS100K cover-song dataset: https://github.com/NovaFrost/SHS100K
