# Soundcheck Backend

Python analysis backend for **Soundcheck** — the AI-music technical-quality scorer (Crate Phase 1). Pairs with the React app at `../quality-scorer/`.

The backend has two entry points sharing one library:

- `backend.cli` — offline ingest, eval, and single-track scoring (`python -m backend.cli {score|ingest|eval}`).
- `backend.api` — FastAPI service exposing `POST /analyze` for live uploads (`uvicorn backend.api:app`).

Both ride on the same internal modules — `signals`, `scoring`, `librosa_engine`, `clap_engine`, `corruptions`, `corpus_io`, `sonics`, `config` — so the offline pass and the live endpoint produce bit-identical results.

The scoring contract is a **bit-equal port** of `../quality-scorer/src/lib/{signals,scoring}.js`. The parity test (`tests/test_scoring_parity.py`) runs the real JS `computeReport` via Node and asserts the Python port matches it exactly.

## Quickstart (dev)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest                      # step 2 gate: signals + parity green
```

Step-by-step build plan lives at the project plan file; locked decisions at `../factory/artifacts/quality-scorer/LOCKED_DECISIONS.md`.
