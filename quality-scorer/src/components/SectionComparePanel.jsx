import { useMemo, useState } from 'react'
import AudioPlayer from './AudioPlayer.jsx'
import NarrativeBlock from './NarrativeBlock.jsx'
import SpectrogramCompare from './SpectrogramCompare.jsx'
import { audioUrlFor } from '../lib/api.js'

/**
 * SectionComparePanel — the expandable panel under each SimilarityRow.
 *
 * Layout (top to bottom):
 *   1. matchTimestamp strip
 *   2. Side-by-side snippet players, both clamped to [startSec, endSec) of
 *      the matched window — always visible (no lazy load) because the
 *      audio playback is the load-bearing primary evidence.
 *   3. Criteria table (tempo / key / harmonic / timbre).
 *   4. Three-tab interface (lazy):
 *      - "Why these are similar"     → NarrativeBlock mode="whySimilar"
 *      - "Make mine more distinctive" → NarrativeBlock mode="creatorAdvice"
 *      - "Visual match"               → SpectrogramCompare
 *
 * When `contextToken` is null (no-key dev env), the two narrative tabs are
 * disabled and a small no-key hint replaces their content. The visual tab
 * always works because it doesn't depend on the backend.
 */
export default function SectionComparePanel({
  matchTimestamp,
  criteria,
  queryFile,
  catalogTrack,
  trackId,
  contextToken,
}) {
  const queryUrl = useMemo(() => {
    if (!queryFile) return null
    try {
      return URL.createObjectURL(queryFile)
    } catch {
      return null
    }
  }, [queryFile])

  const catalogUrl = audioUrlFor(catalogTrack)
  const qs = Number(matchTimestamp?.queryStartSec)
  const qe = Number(matchTimestamp?.queryEndSec)
  const cs = Number(matchTimestamp?.catalogStartSec)
  const ce = Number(matchTimestamp?.catalogEndSec)

  const hasQueryWindow = Number.isFinite(qs) && Number.isFinite(qe) && qe > qs
  const hasCatalogWindow = Number.isFinite(cs) && Number.isFinite(ce) && ce > cs

  // Tab state. Default: whySimilar — the LLM narrative is the headline
  // value-add and the user came here to see it. Tabs lazy-mount: only the
  // currently-active tab's content renders, so NarrativeBlock's fetch only
  // fires on first click.
  const [activeTab, setActiveTab] = useState('whySimilar')
  const narrativeDisabled = !contextToken

  return (
    <div
      className="mt-3 mb-1 rounded-sm p-4"
      style={{
        background: 'var(--color-elev)',
        border: '1px solid var(--color-line)',
      }}
    >
      <div
        className="font-mono text-[11px] tabular-nums"
        style={{ color: 'var(--color-dim)' }}
      >
        {formatTimestamp(matchTimestamp)}
      </div>

      <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
        <SnippetBlock
          label="Your upload"
          src={queryUrl}
          startSec={hasQueryWindow ? qs : null}
          endSec={hasQueryWindow ? qe : null}
          note={
            queryFile
              ? `${fmt(qs)}–${fmt(qe)} of ${truncate(queryFile.name, 36)}`
              : 'upload not available for playback'
          }
        />
        <SnippetBlock
          label="Catalog match"
          src={catalogUrl}
          startSec={hasCatalogWindow ? cs : null}
          endSec={hasCatalogWindow ? ce : null}
          note={
            catalogUrl
              ? `${fmt(cs)}–${fmt(ce)} of ${truncate(catalogTrack?.title || 'track', 36)}`
              : 'preview audio unavailable'
          }
        />
      </div>

      {criteria && (
        <div className="mt-5">
          <div
            className="font-mono text-[11px] uppercase"
            style={{
              color: 'var(--color-faint)',
              letterSpacing: '0.14em',
              fontWeight: 500,
            }}
          >
            Criteria comparison
          </div>
          <div className="mt-2 grid grid-cols-1 gap-1">
            <CriterionRow
              name="Tempo"
              entry={criteria.tempo}
              queryFormat={(v) => `${Math.round(v)} BPM`}
              matchFormat={(v) => `${Math.round(v)} BPM`}
            />
            <CriterionRow
              name="Key"
              entry={criteria.key}
              queryFormat={(v) => String(v)}
              matchFormat={(v) => String(v)}
            />
            <CriterionRow name="Harmonic content" entry={criteria.harmonic} />
            <CriterionRow name="Timbre" entry={criteria.timbre} />
          </div>
        </div>
      )}

      {/* Three-tab RAG explanatory layer + visual match — Commit B. */}
      <div className="mt-5">
        <div
          className="flex gap-0 border-b"
          style={{ borderColor: 'var(--color-line)' }}
        >
          <TabButton
            label="Why these are similar"
            isActive={activeTab === 'whySimilar'}
            isDisabled={narrativeDisabled}
            disabledHint="Narrative disabled — no API key set."
            onClick={() => setActiveTab('whySimilar')}
          />
          <TabButton
            label="Make mine more distinctive"
            isActive={activeTab === 'creatorAdvice'}
            isDisabled={narrativeDisabled}
            disabledHint="Narrative disabled — no API key set."
            onClick={() => setActiveTab('creatorAdvice')}
          />
          <TabButton
            label="Visual match"
            isActive={activeTab === 'visual'}
            isDisabled={false}
            onClick={() => setActiveTab('visual')}
          />
        </div>

        <div className="mt-1">
          {activeTab === 'whySimilar' && (
            narrativeDisabled ? (
              <NarrativeNoKeyFallback />
            ) : (
              <NarrativeBlock
                contextToken={contextToken}
                trackId={trackId}
                mode="whySimilar"
                modeLabel="explanation"
              />
            )
          )}
          {activeTab === 'creatorAdvice' && (
            narrativeDisabled ? (
              <NarrativeNoKeyFallback />
            ) : (
              <NarrativeBlock
                contextToken={contextToken}
                trackId={trackId}
                mode="creatorAdvice"
                modeLabel="advice"
              />
            )
          )}
          {activeTab === 'visual' && (
            <SpectrogramCompare
              queryFile={queryFile}
              catalogTrack={catalogTrack}
              matchTimestamp={matchTimestamp}
            />
          )}
        </div>
      </div>
    </div>
  )
}


function TabButton({ label, isActive, isDisabled, disabledHint, onClick }) {
  return (
    <button
      type="button"
      onClick={isDisabled ? undefined : onClick}
      disabled={isDisabled}
      title={isDisabled ? disabledHint : undefined}
      className="px-3 py-2 font-mono text-[11px] uppercase transition-colors"
      style={{
        letterSpacing: '0.12em',
        fontWeight: 500,
        color: isDisabled
          ? 'var(--color-faint)'
          : isActive
          ? 'var(--color-accent)'
          : 'var(--color-dim)',
        borderBottom: isActive ? '2px solid var(--color-accent)' : '2px solid transparent',
        marginBottom: '-1px',
        cursor: isDisabled ? 'not-allowed' : 'pointer',
        opacity: isDisabled ? 0.55 : 1,
      }}
    >
      {label}
    </button>
  )
}


function NarrativeNoKeyFallback() {
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
        Narrative disabled
      </div>
      <p
        className="mt-2 text-[13px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        The explanatory layer is off in this environment (no API key configured).
        The criteria table above and the side-by-side audio above are the
        available evidence. Switch to the "Visual match" tab for the
        spectrogram view.
      </p>
    </div>
  )
}


function SnippetBlock({ label, src, startSec, endSec, note }) {
  return (
    <div
      className="rounded-sm p-3"
      style={{
        background: 'var(--color-bg)',
        border: '1px solid var(--color-line)',
      }}
    >
      <div
        className="font-mono text-[10px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.14em',
          fontWeight: 500,
        }}
      >
        {label}
      </div>
      <div className="mt-2">
        <AudioPlayer src={src} startSec={startSec} endSec={endSec} durationSec={endSec ? endSec - (startSec || 0) : 10} />
      </div>
      <div
        className="mt-2 font-mono text-[10px] tabular-nums"
        style={{ color: 'var(--color-faint)' }}
      >
        {note}
      </div>
    </div>
  )
}


function CriterionRow({ name, entry, queryFormat, matchFormat }) {
  if (!entry) return null
  const agreement = Math.max(0, Math.min(1, Number(entry.agreement) || 0))
  const barPct = Math.round(agreement * 100)
  const hasValues = entry.queryValue != null && entry.matchValue != null
  return (
    <div
      className="grid items-center gap-3 py-1.5"
      style={{
        gridTemplateColumns: '110px 1fr 56px 1fr 200px',
        fontSize: '12px',
      }}
    >
      <span style={{ color: 'var(--color-dim)' }}>{name}</span>
      <span
        className="font-mono tabular-nums"
        style={{ color: 'var(--color-ink)' }}
      >
        {hasValues ? (queryFormat ? queryFormat(entry.queryValue) : entry.queryValue) : ''}
      </span>
      <span
        className="text-center font-mono text-[11px]"
        style={{ color: 'var(--color-faint)' }}
      >
        ↔
      </span>
      <span
        className="font-mono tabular-nums"
        style={{ color: 'var(--color-ink)' }}
      >
        {hasValues ? (matchFormat ? matchFormat(entry.matchValue) : entry.matchValue) : ''}
      </span>
      <span className="flex items-center gap-2">
        <span
          className="block h-1 flex-1 rounded-sm"
          style={{ background: 'var(--color-line)' }}
        >
          <span
            className="block h-full rounded-sm"
            style={{ width: `${barPct}%`, background: 'var(--color-accent)' }}
          />
        </span>
        <span
          className="shrink-0 font-mono text-[10px]"
          style={{ color: 'var(--color-faint)' }}
          title={`agreement ${entry.agreement?.toFixed?.(2) ?? agreement.toFixed(2)}`}
        >
          {entry.label}
        </span>
      </span>
    </div>
  )
}


function formatTimestamp(ts) {
  if (!ts) return ''
  return `match: query ${fmt(ts.queryStartSec)}–${fmt(ts.queryEndSec)} ↔ track ${fmt(ts.catalogStartSec)}–${fmt(ts.catalogEndSec)}`
}


function fmt(sec) {
  const n = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(n / 60)
  const r = n % 60
  return `${m}:${r.toString().padStart(2, '0')}`
}


function truncate(s, n) {
  if (!s) return ''
  return s.length > n ? `${s.slice(0, n - 1)}…` : s
}
