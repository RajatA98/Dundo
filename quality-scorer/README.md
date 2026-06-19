# Dundo — frontend

React + Vite + Tailwind v4 — the upload flow, ReportCard (similarity-first), and `/evaluation` page.

> The directory is named `quality-scorer/` for legacy reasons (this codebase started as a separate quality scorer before pivoting to similarity). Renaming would touch Vercel config and import paths for marginal benefit; the directory name stays.

The backend lives at `../backend/` and exposes `POST /neighbors` (the similarity report) and `POST /analyze` (the inherited quality badge). Both are consumed by `src/lib/api.js`.

## Run

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production bundle → dist/
npm run preview  # serve the built bundle
```

Set `VITE_API_URL` to the backend host:

- Dev: `http://localhost:8000` (in `.env.local`)
- Prod: the deployed HF Space URL (in `.env.production`)

## Structure

```
src/
  lib/
    api.js              # neighborsUpload + analyzeUpload; the only seam to the backend
    format.js, prng.js  # formatting + deterministic RNG helpers
  components/
    Nav.jsx, Layout.jsx, Hero.jsx
    DropZone.jsx, ExampleChips.jsx
    ReportCard.jsx                # Case A + Case B in one component
    SimilarityReport.jsx          # top-3 ranked rows + headline
    AcrCloudRow.jsx, SunoPill.jsx # two ACRCloud rows; rose tint when likely_source == "suno"
    QualityBadge.jsx              # inline badge + expandable 7-signal breakdown
  pages/
    ScorerPage.jsx       # landing — drop zone, examples, ReportCard
    EvaluationPage.jsx   # measured detector quality + named FP/FN examples
    AboutPage.jsx
```

The design tokens live in `tailwind.config.js`. The Suno-flare tokens (`--suno`, `--suno-soft`, `--suno-deep`) are only consumed by `SunoPill.jsx` and a small detector sigil in the footer — see `factory/artifacts/CLAUDE_UI_DESIGN_PROMPT.md` for the rules.

## Stack

React 18 + Vite + Tailwind v4 + Framer Motion (transitions only; no springs).
