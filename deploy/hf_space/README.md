---
title: Dundo
emoji: 🎵
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Dundo API

FastAPI service that takes an AI-generated music track, encodes it to a 512-d
MuQ-MuLan music-text joint embedding (10s windowed, L2-normalized mean pool),
and returns the **top-K closest indie artists in the Creative-Commons-licensed
catalog** ranked by cosine similarity. A separate `/narrative` endpoint
generates a grounded LLM explanation of *why* each match resonates with the
upload. A legacy `/analyze` endpoint preserved from the prior quality-detector
pipeline returns a 7-signal librosa-based brokenness report.

This is the backend half of the project. The React frontend on Vercel calls
this service.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | `{"ok": true, "model": "...", "corpus": <N>}` |
| `POST` | `/neighbors?k=5` | multipart `file=...` → top-K matches with similarity metrics, criteria comparison block, and a signed `contextToken` for the narrative endpoint |
| `POST` | `/narrative` | JSON `{contextToken, trackId, mode}` → LLM-grounded "why this resonates" narrative with structured citations |
| `GET` | `/narrative/stats` | in-process telemetry counters for the narrative layer |
| `POST` | `/analyze` | multipart `file=...` → legacy 7-signal quality report |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CORS_ORIGIN` | `http://localhost:5173` | Frontend origin allowed in addition to the `*.vercel.app` regex. Set this to your Vercel production URL. |
| `PORT` | `7860` | Server bind port (HF Spaces provides this). |
| `HF_HOME` | `/app/.hf_cache` | Model cache location (set in the Dockerfile). |
| `CORPUS_DIR` | (auto) | Override the corpus directory. Defaults to `/app/quality-scorer/public/corpus`. |
| `SIMILARITY_THRESHOLD_DEFAULT` | `0.70` | Below this cosine, the frontend renders the style-attribution / empty state. |
| `SIMILARITY_BACKEND` | `numpy` | `numpy` (default) or `faiss` to use the FAISS Flat backend behind the same `top_k_neighbors` contract. Set to `faiss` once the catalog crosses ~10K tracks. |
| `OPENAI_API_KEY` | — | Powers the `/narrative` discovery-explanation layer. When unset, `/narrative` returns `503 narrative-disabled` and the frontend tabs render the no-key fallback. `/neighbors` is unaffected. |
| `CONTEXT_TOKEN_HMAC_KEY` | — | Signs the opaque `contextToken` that `/neighbors` attaches. Generate with `openssl rand -hex 32`. Without it, `/narrative` also returns `503 narrative-disabled`. |
| `OPENAI_MODEL_ID` | `gpt-4o-mini` | Optional override for the LLM model id. |
| `BANDSINTOWN_APP_ID` | — | (Coming) Powers the live-show row on each discovery card; nullable — cards without show data simply don't render that row. |

Set these via the Space's **Settings → Variables and secrets** tab. The keys
must be marked as Secret (not Variable) so they are not echoed in build logs.

## Catalog rights

The reference corpus is **Creative-Commons-licensed indie music** from
MTG-Jamendo (today) and Free Music Archive (planned bulk expansion). Each
match links out to the artist's source page (Jamendo / FMA / Bandcamp) for
attribution and so the discovery can turn into action.

There is no commercial-catalog ingestion. The PiedPiper-era iTunes Tier-1
ingest + ACRCloud commercial second-opinion signals were retired in the
Dundo pivot — see `factory/artifacts/ACRCLOUD_RETIREMENT_NOTE.md` and the
top-level `JOURNEY.md` for the rationale.

## Cold start

Free CPU Basic Spaces sleep after ~48 h idle and take ~30 s to wake on the
first request. The Dundo frontend handles this with a "warming up the
analyzer" UI state when a request exceeds 6 s. An UptimeRobot ping on
`/health` every 5 minutes keeps the Space warm during the demo window — setup
is documented in the top-level repo README.
