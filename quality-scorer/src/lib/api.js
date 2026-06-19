// Single seam between the React app and the Python backend.
//
// `VITE_API_URL` bakes into the bundle at build time:
//   - Dev (.env.local):       http://localhost:8000
//   - Prod (.env.production): https://<your-hf-username>-piedpiper.hf.space
//
// If unset, falls back to relative `/analyze` (same-origin) — useful for the
// all-in-one HF Space deploy where the static site is served by the API host.
export const API_BASE = import.meta.env.VITE_API_URL || ''

/**
 * POST a File to `/analyze` and return the full Track-shape JSON.
 * Throws Error(message) on non-2xx; the page maps that to the `ErrorState` UI.
 *
 * Phase 3+: `/analyze` is retained for the inherited 7-signal quality badge.
 * The new headline (similarity) flow uses `neighborsUpload` below.
 */
export async function analyzeUpload(file) {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: fd })
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = body?.error || ''
    } catch {
      /* not json */
    }
    throw new Error(detail || `HTTP ${r.status}`)
  }
  return r.json()
}

/**
 * POST a File to `/neighbors` and return the similarity report.
 *
 * Phase 2 backend response shape (locked):
 * {
 *   query: { ... },                          // Track-shape echo of the upload (id, title, durationSec, ...)
 *   neighbors: [
 *     {
 *       trackId: string,                     // e.g. "tier1:itunes:1499378034"
 *       meanPooledSimilarity: number,        // cosine [-1, 1], the ranking signal
 *       maxSegmentSimilarity: number,        // cosine [-1, 1], local-resemblance secondary
 *       track: { title, artist, source, ... } // catalog metadata attached server-side
 *     },
 *     ...                                    // length min(k, N), sorted by meanPooledSimilarity desc
 *   ],
 *   topMeanPooledSimilarity: number,         // == neighbors[0].meanPooledSimilarity for convenience
 *   topMaxSegmentSimilarity: number,         // == neighbors[0].maxSegmentSimilarity
 *   modelSha: string,                        // pinned MuQ-MuLan revision SHA from manifest.json (ADR-0002)
 *   thresholdDefault: number,                // "Completely unique" cutoff (provisional 0.70)
 *   // OR, when the catalog isn't loaded:
 *   verdict: "no_corpus",
 *   neighbors: []
 * }
 *
 * Frontend applies the threshold rule: if topMeanPooledSimilarity >= thresholdDefault
 * → render Case A headline as `{similarityLabel} · {percentileRank}th percentile match`
 * (per ADR-0001 — raw cosine is shown small in technical detail, not converted to a
 * percent and not phrased as a copyright/infringement number).
 * Otherwise → render Case B (`"Completely unique — this track doesn't sound like
 * anything in our reference catalog"`).
 *
 * @param {File} file - the audio file to analyze (mp3/wav/flac/ogg/m4a, ≤50MB)
 * @param {number} [k=5] - number of neighbors to return
 * @returns {Promise<object>} the neighbors response (see shape above)
 * @throws {Error} on non-2xx with the backend's `error` field as the message
 */
export async function neighborsUpload(file, k = 5) {
  const fd = new FormData()
  fd.append('file', file)
  const qs = k === 5 ? '' : `?k=${encodeURIComponent(k)}`
  const r = await fetch(`${API_BASE}/neighbors${qs}`, {
    method: 'POST',
    body: fd,
  })
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = body?.error || ''
    } catch {
      /* not json */
    }
    throw new Error(detail || `HTTP ${r.status}`)
  }
  return r.json()
}

/**
 * Pull the artwork URL out of a catalog track, scaled to the requested size.
 *
 * iTunes URLs end with `/100x100bb.jpg` — we can request larger by string-replace.
 * Jamendo URLs (added during the Phase 7.5 enrichment) come pre-sized at 300x300.
 *
 * Returns null when no artwork is available (renders a placeholder tile).
 */
export function artworkUrlFor(track, size = 100) {
  if (!track) return null
  const url = track.artwork_url ?? track.artworkUrl ?? null
  if (!url) return null
  // iTunes pattern — replace the trailing /NNNxNNNbb.jpg with desired size.
  return url.replace(/\d+x\d+bb\.jpg$/, `${size}x${size}bb.jpg`)
}

/**
 * Pull the playable audio URL out of a catalog track, if any.
 *
 * Returns:
 *   - string URL  → for iTunes (previewUrl, 30s m4a) or Jamendo (audioStreamUrl from enrichment)
 *   - null        → no playable audio for this track (renders the play button disabled)
 *
 * Pure function. Component-agnostic. Future sources just add a new key here.
 */
export function audioUrlFor(track) {
  if (!track) return null
  const ext = track.external_ids ?? track.externalIds ?? {}
  return (
    ext.previewUrl
    ?? ext.jamendoAudioUrl
    ?? ext.jamendoStreamUrl
    ?? track.preview_url
    ?? null
  )
}

/**
 * POST to `/narrative` and return the typed RAG narrative result.
 *
 * ADR-0005 (Commit C) explanatory layer on top of /neighbors retrieval.
 * The backend lazy-imports OpenAI's GPT-4o-mini and validates the LLM's
 * structured citations against the per-neighbor context embedded inside
 * the signed `contextToken` /neighbors issued at retrieval time.
 *
 * Returns a discriminated union by `kind`:
 *   - `{kind: "narrative", mode, prose, citations: [...]}` → success
 *   - `{kind: "low_confidence", reason: "..."}`            → gate short-circuited the LLM
 *   - `{kind: "unavailable", reason: "..."}`               → LLM produced something unusable
 *
 * On HTTP error (4xx/5xx) throws `Error(code)` where `code` is the typed
 * backend error string. UI components map these to specific fallback panels:
 *   - "narrative-disabled" (503) → no OPENAI_API_KEY or HMAC key (no-key fallback)
 *   - "token-expired"     (412) → contextToken aged out (TTL = 30 min)
 *   - "stale-token"       (412) → catalog or model SHA changed since /neighbors
 *   - "invalid-token"     (401) → tampered or wrong-secret token
 *   - "malformed-token"   (400) → bad shape (shouldn't happen for valid clients)
 *   - "not-in-context"    (404) → trackId wasn't in the token's allowlist
 *   - "unsupported-mode"  (422) → mode not in {"whySimilar", "creatorAdvice"}
 *   - "malformed-context" (422) → token decoded but context fragment failed validation
 *   - "narrative-error"   (500) → unexpected backend failure
 *
 * @param {string} contextToken - opaque token from /neighbors response
 * @param {string} trackId      - which neighbor to narrate
 * @param {"whySimilar"|"creatorAdvice"} mode
 * @returns {Promise<object>} the typed result with .kind discriminator
 * @throws {Error} on non-2xx with the backend's `error` field as message
 */
export async function fetchNarrative(contextToken, trackId, mode) {
  const t0 = performance.now()
  if (!contextToken) {
    _narrativeBreadcrumb({ level: 'warning', mode, kind: 'error', code: 'narrative-disabled' })
    throw new Error('narrative-disabled')
  }
  let r
  try {
    r = await fetch(`${API_BASE}/narrative`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contextToken, trackId, mode }),
    })
  } catch (networkErr) {
    _narrativeBreadcrumb({ level: 'error', mode, kind: 'error', code: 'network-error' })
    throw new Error('network-error')
  }
  if (!r.ok) {
    let code = ''
    try {
      const body = await r.json()
      code = body?.error || ''
    } catch {
      /* not json */
    }
    const latencyMs = Math.round(performance.now() - t0)
    _narrativeBreadcrumb({
      level: r.status >= 500 ? 'error' : 'warning',
      mode,
      kind: 'error',
      code: code || `http-${r.status}`,
      latencyMs,
    })
    throw new Error(code || `HTTP ${r.status}`)
  }
  const body = await r.json()
  const latencyMs = Math.round(performance.now() - t0)
  _narrativeBreadcrumb({
    level: body.kind === 'narrative' ? 'info' : 'warning',
    mode,
    kind: body.kind || 'unknown',
    code: body.reason || null,
    latencyMs,
  })
  return body
}


/**
 * Drop a Sentry breadcrumb describing a narrative call outcome. No-op when
 * Sentry isn't on the window (dev environment, no DSN). Stays cheap: one
 * imported function call per /narrative call, no global mutation.
 */
async function _narrativeBreadcrumb({ level, mode, kind, code, latencyMs }) {
  try {
    // Dynamic-import @sentry/react so this module stays Sentry-agnostic.
    // The bundle already includes Sentry from the app's existing wiring,
    // so the import is free.
    const Sentry = await import('@sentry/react')
    Sentry.addBreadcrumb({
      category: 'narrative',
      level,
      message: `narrative.${kind}`,
      data: {
        mode,
        kind,
        code: code || undefined,
        latencyMs: latencyMs ?? undefined,
      },
    })
  } catch {
    /* Sentry not installed / not configured — no observability, no error */
  }
}

/**
 * Apply the locked threshold rule to a /neighbors response.
 * Returns the calibrated display headline per ADR-0001.
 *
 * Returns { caseA, topPercentile, topLabel, topRawCosine, topSegment, topMatch, querySpecificity }.
 * `topPct` is preserved as an alias for `topPercentile * 100` for any caller
 * that still wants a 0-100 number (e.g., bar widths).
 *
 * Pure function — components consume it for headline rendering without
 * re-encoding the threshold logic per-component.
 */
export function deriveHeadline(response) {
  if (!response || response.verdict === 'no_corpus' || !response.neighbors?.length) {
    return {
      caseA: false,
      topPct: null,
      topPercentile: null,
      topLabel: null,
      topRawCosine: null,
      topSegment: null,
      topMatch: null,
      querySpecificity: null,
    }
  }
  const top = response.neighbors[0]
  const rawCosine = top.rawCosine ?? response.topMeanPooledSimilarity ?? top.meanPooledSimilarity ?? 0
  const percentile = top.percentileRank ?? response.topPercentileRank ?? null
  const label = top.similarityLabel ?? response.topSimilarityLabel ?? null
  const segment = top.segmentSupport ?? top.maxSegmentSimilarity ?? null
  const threshold = response.thresholdDefault ?? 0.70
  return {
    caseA: rawCosine >= threshold,
    topPercentile: percentile,
    topLabel: label,
    topRawCosine: rawCosine,
    topSegment: segment,
    topPct: Math.round(rawCosine * 1000) / 10,  // legacy alias
    topMatch: top,
    querySpecificity: response.querySpecificity ?? null,
  }
}

/**
 * Format a percentile rank [0, 1] as the visible UI string.
 * 0.992 -> "99th percentile"; 0.503 -> "50th percentile"; 0.04 -> "4th percentile".
 *
 * Returns null when the percentile is null (no calibration available yet).
 */
export function fmtPercentile(p) {
  if (p == null || Number.isNaN(p)) return null
  const n = Math.max(0, Math.min(100, Math.round(p * 100)))
  if (n === 0) return '<1st percentile'
  const suffix = (() => {
    const lastTwo = n % 100
    if (lastTwo >= 11 && lastTwo <= 13) return 'th'
    const last = n % 10
    if (last === 1) return 'st'
    if (last === 2) return 'nd'
    if (last === 3) return 'rd'
    return 'th'
  })()
  return `${n}${suffix} percentile`
}
