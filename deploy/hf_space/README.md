---
title: PiedPiper
emoji: 🎵
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PiedPiper API

FastAPI service that takes an AI-generated music track, encodes it to a 512-d
LAION-CLAP music-tuned audio embedding (10s windowed, L2-normalized mean pool),
and returns the **top-K closest tracks in a 160-track reference catalog** ranked
by cosine similarity. A legacy `/analyze` endpoint preserved from the prior
quality-detector pipeline returns a 7-signal librosa-based brokenness report.

Two independent secondary signals from ACRCloud — **Cover Song ID** and
**AI Music Detector** — are exposed as additional `/neighbors` response fields
behind the `ENABLE_ACRCLOUD` flag.

This is the backend half of the project. The React frontend on Vercel calls this
service.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | `{"ok": true, "model": "...", "corpus": <N>}` |
| `POST` | `/neighbors?k=3` | multipart `file=...` → top-K matches with `meanPooledSimilarity` + `maxSegmentSimilarity` + optional ACRCloud signals |
| `POST` | `/analyze` | multipart `file=...` → legacy 7-signal quality report |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CORS_ORIGIN` | `http://localhost:5173` | Frontend origin allowed in addition to the `*.vercel.app` regex. Set this to your Vercel production URL. |
| `PORT` | `7860` | Server bind port (HF Spaces provides this). |
| `HF_HOME` | `/app/.hf_cache` | Model cache location (set in the Dockerfile). |
| `CORPUS_DIR` | (auto) | Override the corpus directory. Defaults to `/app/quality-scorer/public/corpus`. |
| `SIMILARITY_THRESHOLD_DEFAULT` | `0.70` | Below this cosine, the frontend renders the "completely unique" empty state. |
| `ENABLE_ACRCLOUD` | `false` | Master gate for both ACRCloud signals. |
| `ACRCLOUD_ACCESS_KEY` | — | Cover Song ID HMAC access key. |
| `ACRCLOUD_ACCESS_SECRET` | — | Cover Song ID HMAC secret. |
| `ACRCLOUD_HOST` | `identify-eu-west-1.acrcloud.com` | Cover Song ID identification host. |
| `ACRCLOUD_AI_DETECTOR_URL` | — | AI Music Detector endpoint URL. |
| `ACRCLOUD_AI_DETECTOR_BEARER` | — | AI Music Detector bearer token. |

Set these via the Space's **Settings → Variables and secrets** tab. The
`ACRCLOUD_*` secrets must be marked as Secret (not Variable) so they are not
echoed in build logs.

## Catalog rights

The reference corpus blends two free-licensable tiers:

- **Tier 1** — iTunes Search API previews. Per Apple terms the preview audio is
  streamed at request time, never cached locally. The catalog stores only metadata
  + the precomputed 512-d embedding. Each Tier-1 row carries `attributionRequired: true`
  and an attribution `trackViewUrl` link-out.
- **Tier 2** — MTG-Jamendo (CC-BY) loaded from the public mirror, normalized to
  the same embedding pipeline. Artist names are anonymized in metadata per the
  Jamendo distribution conventions.

The eval framing (`/evaluation` page) names this catalog composition explicitly
as a known limitation.

## Cold start

Free CPU Basic Spaces sleep after ~48 h idle and take ~30 s to wake on the first
request. The PiedPiper frontend handles this with a "warming up the analyzer" UI
state when a request exceeds 6 s. An UptimeRobot ping on `/health` every 5 minutes
keeps the Space warm during the demo window — setup is documented in the top-level
repo README.
