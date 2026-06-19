# Credits

Dundo is forked from [PiedPiper](https://github.com/RajatA98/PiedPiper) at commit `c04f38e` (2026-06-19). The engineering substance carries over wholesale; the positioning, catalog policy, and one external dependency change.

## What PiedPiper contributed to Dundo

| Component | Where it came from |
|---|---|
| MuQ-MuLan 512-d retrieval pipeline + the LAION-CLAP → MuQ-MuLan swap | PiedPiper ADR-0002 |
| Density-relative calibration argument for catalog scale | PiedPiper ADR-0003 |
| Four-criterion similarity layer (tempo / key / harmonic / timbre) | PiedPiper ADR-0004 |
| RAG narrative layer with HMAC-signed context tokens + structured citation validation | PiedPiper ADR-0005 |
| FAISS Flat backend behind `SIMILARITY_BACKEND` flag + benchmark script | PiedPiper commit `c138309` |
| Verification harness (`backend/backend/scripts/verify_matching.py`) | PiedPiper Phase 6 |
| 12-case RAG eval harness with five baseline gates | PiedPiper Phase 6 |
| Three-tab match-explanation UI (Why / Distinctive / Visual) | PiedPiper Phase 7 |
| Lazy WaveSurfer.js spectrogram tab | PiedPiper Phase 7 |
| Deployment pattern (HF Space + Vercel + UptimeRobot) | PiedPiper Phase 5 |
| In-process telemetry + `/narrative/stats` endpoint | PiedPiper Phase 7 |
| The 177-backend / 20-frontend test suite | PiedPiper Phases 1–7 |

## What changed in the Dundo fork

| Change | Rationale |
|---|---|
| **New name + positioning**: "discover indie artists similar to your AI music" replaces "acoustic-similarity scanner for AI-generated music" | The defensive copyright-detector framing was off-thesis for a Suno-targeted pitch; discovery is positive-sum. See `JOURNEY.md`. |
| **ACRCloud removed** | See `factory/artifacts/ACRCLOUD_RETIREMENT_NOTE.md`. ACRCloud's commercial second-opinion catalog answers "does this resemble a known commercial composition" — a defensive question Dundo does not ask. |
| **iTunes Tier-1 ingest removed** | The Apple iTunes Search API preview-fetch sat on a knife's edge of preview-terms compliance per PiedPiper ADR-0002. Dundo's CC-only catalog story is cleaner without it. |
| **Bandsintown integration** (planned) | Live-show data per discovered artist — turns discovery into actionable support. |
| **README + frontend reframed** | Discovery-positive copy replaces risk-verdict copy throughout. |

## Why a fork instead of a rename

Per Codex's pre-fork review (`factory/artifacts/CODEX_DUNDO_PIVOT_REVIEW_RESPONSE.md`): clean git history for Dundo from commit 1 reads better to a reviewer than an in-place rename PR. PiedPiper stays open as engineering provenance and the historical audit trail of every decision that led here.

## Acknowledgments

- **Tencent AI Lab** for [MuQ-MuLan](https://github.com/tencent-ailab/MuQ) (CC-BY-NC 4.0).
- **MTG-Jamendo** ([repo](https://github.com/MTG/mtg-jamendo-dataset), [paper](https://repositori.upf.edu/handle/10230/42015)) for the Creative-Commons indie music catalog.
- **Free Music Archive** ([repo](https://github.com/mdeff/fma), [paper](https://arxiv.org/abs/1612.01840)) for the bulk-ingestible CC music dataset Dundo will scale into next.
- **OpenAI** for GPT-4o-mini, the LLM that powers the `/narrative` discovery-explanation layer.
- **librosa**, **soundfile**, **FastAPI**, **React**, **WaveSurfer.js**, **Sentry**, **Vercel**, **Hugging Face Spaces** — the open infrastructure Dundo runs on.
- **Codex (GPT-5)** and **Perplexity** for the engineering + market reviews of the PiedPiper → Dundo pivot.
