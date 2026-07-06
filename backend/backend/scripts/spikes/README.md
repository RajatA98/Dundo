# Validation spikes

One-off scripts that validated a specific design decision before it shipped.
Kept for provenance (they're the evidence behind a real number or a real
"rejected because" in an ADR/decision doc) — not maintained as production
code, not run in CI, not imported by anything except `evidence_sweep.py`
(which reads `SUPER_GENRE` from `build_catalog_tags.py`).

- `evidence_eval_spike.py`, `evidence_eval_spike2.py`, `evidence_eval_knn.py`
  — validated the Evidence Layer's tag-recovery approach (k-NN propagation
  of real MTG-Jamendo tags vs. zero-shot prediction from MuQ embeddings).
  See `evidence_tags.py` and `factory/artifacts/CODEX_EVIDENCE_LAYER_*.md`.
- `evidence_sweep.py` — the tau=0.30 sweep cited directly in
  `evidence_tags.py`'s module docstring.
- `muq_tag_spike.py` — early spike for tagging catalog tracks from MuQ
  embeddings directly; superseded by `build_catalog_tags.py`, which instead
  joins the catalog to real MTG-Jamendo editorial tags.
- `mb_coverage_spike.py` — the MusicBrainz coverage check (42% match rate)
  that led to rejecting MusicBrainz as a narrative-grounding source in favor
  of the catalog's own real tags — see the "Knowledge-narrative" decision.
