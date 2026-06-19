import { useEffect, useState } from 'react'
import { fetchNarrative } from '../lib/api.js'

/**
 * NarrativeBlock — renders the typed RAG narrative response from /narrative
 * for ONE mode ("whySimilar" or "creatorAdvice"). Mounted lazily inside a
 * tab; on first mount it fires fetchNarrative() and renders the result
 * (or one of several typed fallback panels) when it arrives.
 *
 * State machine (drives the visible UI):
 *   - idle      — initial, before first fetch fires
 *   - loading   — fetch in-flight
 *   - success   — backend returned a typed body with .kind discriminator
 *   - error     — backend returned a non-2xx; .errorCode is the typed code
 *
 * The success branch further branches on the .kind discriminator:
 *   - "narrative"      — prose + citation chips
 *   - "low_confidence" — gate short-circuited; render typed copy + cta
 *   - "unavailable"    — LLM output rejected by validation; render error tone
 *
 * Citations render trackId in the chip tooltip (Codex round-1 Q5 fix) so
 * ambiguous titles disambiguate without UI clutter. Visible chip text is
 * `<criterion ids> · <side> <ts>` or `cosine` when rawCosine is cited.
 */
export default function NarrativeBlock({ contextToken, trackId, mode, modeLabel }) {
  const [state, setState] = useState('idle')
  const [data, setData] = useState(null)
  const [errorCode, setErrorCode] = useState(null)

  useEffect(() => {
    let cancelled = false
    setState('loading')
    setData(null)
    setErrorCode(null)
    fetchNarrative(contextToken, trackId, mode)
      .then((body) => {
        if (cancelled) return
        setData(body)
        setState('success')
      })
      .catch((err) => {
        if (cancelled) return
        setErrorCode(err?.message || 'unknown-error')
        setState('error')
      })
    return () => {
      cancelled = true
    }
  }, [contextToken, trackId, mode])

  if (state === 'loading') {
    return <NarrativeSkeleton modeLabel={modeLabel} />
  }
  if (state === 'error') {
    return <NarrativeError code={errorCode} />
  }
  if (state === 'success' && data) {
    if (data.kind === 'narrative') {
      return <NarrativeSuccess data={data} />
    }
    if (data.kind === 'low_confidence') {
      return <NarrativeLowConfidence reason={data.reason} />
    }
    if (data.kind === 'unavailable') {
      return <NarrativeUnavailable reason={data.reason} />
    }
  }
  // idle — should be a fleeting state since useEffect fires the call
  return <NarrativeSkeleton modeLabel={modeLabel} />
}


function NarrativeSkeleton({ modeLabel }) {
  return (
    <div className="space-y-2 px-1 py-3">
      <div
        className="font-mono text-[10px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.14em',
          fontWeight: 500,
        }}
      >
        Generating {modeLabel || 'narrative'}…
      </div>
      <div
        className="h-2 w-full rounded-sm animate-pulse"
        style={{ background: 'var(--color-line)' }}
      />
      <div
        className="h-2 w-11/12 rounded-sm animate-pulse"
        style={{ background: 'var(--color-line)' }}
      />
      <div
        className="h-2 w-3/4 rounded-sm animate-pulse"
        style={{ background: 'var(--color-line)' }}
      />
    </div>
  )
}


function NarrativeSuccess({ data }) {
  const prose = data.prose || ''
  const citations = Array.isArray(data.citations) ? data.citations : []
  return (
    <div className="px-1 py-3">
      <p
        className="text-[13px] leading-relaxed"
        style={{ color: 'var(--color-ink)' }}
      >
        {prose}
      </p>
      {citations.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {citations.map((c, i) => (
            <CitationChip key={i} citation={c} />
          ))}
        </div>
      )}
    </div>
  )
}


function CitationChip({ citation }) {
  const ids = Array.isArray(citation.criterionIds) ? citation.criterionIds : []
  const side = citation.side === 'match' ? 'match' : 'query'
  const range = Array.isArray(citation.timestampRange) ? citation.timestampRange : null
  const tsLabel = range
    ? `${fmtSec(range[0])}–${fmtSec(range[1])}`
    : null

  const tooltip = [
    citation.trackId ? `trackId: ${citation.trackId}` : null,
    ids.length ? `criteria: ${ids.join(', ')}` : null,
    citation.citedValues ? `values: ${formatCitedValues(citation.citedValues)}` : null,
  ]
    .filter(Boolean)
    .join('\n')

  const display = ids.length ? ids.join(' · ') : 'evidence'
  const sideSuffix = tsLabel ? ` · ${side} ${tsLabel}` : ` · ${side}`

  return (
    <span
      title={tooltip}
      className="inline-flex items-center gap-1 rounded-sm px-2 py-0.5 font-mono text-[10px]"
      style={{
        background: 'var(--color-elev)',
        border: '1px solid var(--color-line)',
        color: 'var(--color-dim)',
      }}
    >
      <span style={{ color: 'var(--color-ink)' }}>{display}</span>
      <span style={{ color: 'var(--color-faint)' }}>{sideSuffix}</span>
    </span>
  )
}


function NarrativeLowConfidence({ reason }) {
  const copy = LOW_CONFIDENCE_COPY[reason] || LOW_CONFIDENCE_COPY['default']
  return (
    <div className="px-1 py-3">
      <div
        className="font-mono text-[10px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.14em',
          fontWeight: 500,
        }}
      >
        Low confidence
      </div>
      <p
        className="mt-2 text-[13px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        {copy}
      </p>
    </div>
  )
}


function NarrativeUnavailable({ reason }) {
  const copy = UNAVAILABLE_COPY[reason] || UNAVAILABLE_COPY['default']
  return (
    <div className="px-1 py-3">
      <div
        className="font-mono text-[10px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.14em',
          fontWeight: 500,
        }}
      >
        Narrative unavailable
      </div>
      <p
        className="mt-2 text-[13px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        {copy}
      </p>
    </div>
  )
}


function NarrativeError({ code }) {
  const copy = ERROR_COPY[code] || ERROR_COPY['default']
  return (
    <div className="px-1 py-3">
      <div
        className="font-mono text-[10px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.14em',
          fontWeight: 500,
        }}
      >
        {ERROR_HEADLINE[code] || ERROR_HEADLINE['default']}
      </div>
      <p
        className="mt-2 text-[13px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        {copy}
      </p>
    </div>
  )
}


const LOW_CONFIDENCE_COPY = {
  'missing-criteria':
    'Not enough computed criteria for this match to support a useful explanation. The cosine similarity is the primary signal here — listen to both snippets above.',
  'missing-metadata':
    'Track metadata is incomplete for this match. The explanatory layer skipped this one to avoid speculation.',
  'weak-evidence':
    'The criteria agreement is too low to ground a confident explanation. The cosine score may still be meaningful — try the side-by-side snippets to judge.',
  'context-cap-exceeded':
    'The structured context exceeded the prompt-size cap. Falling back to the criteria block above.',
  default:
    'Confidence is too low to generate a useful narrative for this match.',
}


const UNAVAILABLE_COPY = {
  'malformed-llm-output':
    'The model returned a response that did not match the expected schema. No explanation rendered to avoid presenting unstructured text.',
  'openai-error':
    'The model could not be reached. Try expanding the row again in a moment.',
  'citation-hallucinated':
    'The model cited criterion values that were not in the supplied context — likely a hallucination. The response was discarded.',
  default:
    'The narrative could not be generated for this match.',
}


const ERROR_HEADLINE = {
  'narrative-disabled': 'Narrative disabled',
  'token-expired': 'Session expired',
  'stale-token': 'Catalog updated',
  'invalid-token': 'Token mismatch',
  'malformed-token': 'Token malformed',
  'not-in-context': 'Track not in context',
  'unsupported-mode': 'Unsupported mode',
  'malformed-context': 'Context invalid',
  'narrative-error': 'Backend error',
  default: 'Error',
}


const ERROR_COPY = {
  'narrative-disabled':
    'The narrative layer is not enabled in this environment. The retrieval bar chart and the side-by-side snippets are the available evidence here.',
  'token-expired':
    'The match context expired (30-minute session). Re-upload the file to regenerate.',
  'stale-token':
    'The catalog or model changed since you uploaded. Re-upload to refresh the context.',
  'invalid-token':
    'The match context could not be verified. Re-upload the file.',
  'malformed-token':
    'The match context is malformed. Re-upload the file to recover.',
  'not-in-context':
    'This track was not part of the original match set. Re-upload to regenerate.',
  'unsupported-mode':
    'This narrative mode is not supported.',
  'malformed-context':
    'The match context could not be reconstructed. Re-upload to recover.',
  'narrative-error':
    'The backend hit an unexpected error generating the narrative. The criteria table above is unaffected.',
  default:
    'Something went wrong generating the narrative. The criteria table above is unaffected.',
}


function fmtSec(sec) {
  const n = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(n / 60)
  const r = n % 60
  return `${m}:${r.toString().padStart(2, '0')}`
}


function formatCitedValues(values) {
  if (!values || typeof values !== 'object') return ''
  return Object.entries(values)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
    .join(', ')
}
