/**
 * SimilarityReport — the headline block: top match + top-3 list, or "Completely unique".
 *
 * The single most important visual artifact in the app. Renders Case A or Case B
 * based on the `caseA` boolean (derived in api.js → deriveHeadline).
 *
 * Visual contract: ui_mockup_v2_suno_flare.html `#report-A` (Case A) and
 * `#report-B` (Case B) blocks.
 *
 * Case A — top match crosses the threshold:
 *   TOP MATCH
 *   87%   similar to
 *   Blinding Lights — The Weeknd                          [↗ open in iTunes]
 *
 *   TOP 3 CLOSEST IN CATALOG
 *   1. Blinding Lights — The Weeknd      ███████░  87%
 *   2. Save Your Tears — The Weeknd      █████░░░  72%
 *   3. Take On Me — a-ha                 ████░░░░  68%
 *
 * Case B — no match crosses the threshold:
 *   RESULT
 *   Completely unique
 *   No close matches in our reference catalog
 *
 *   CLOSEST TRACKS · FOR REFERENCE, NOT MATCHES
 *   1. Pompeii — Bastille                 ███░░░░░  58%   (muted)
 *   ...
 *
 * The Case B empty-state copy is LOCKED — it must match the string in
 * LOCKED_DECISIONS exactly: "Completely unique — this track doesn't sound
 * like anything in our reference catalog".
 *
 * @param {Object} props
 * @param {boolean} props.caseA         - true → Case A, false → Case B
 * @param {Array}   props.neighbors     - response.neighbors (already top-3 sliced)
 * @param {number}  props.topPct        - rounded headline percentage for Case A
 */
import { useState } from 'react'
import SimilarityRow from './SimilarityRow.jsx'
import AudioPlayer from './AudioPlayer.jsx'
import { audioUrlFor, artworkUrlFor } from '../lib/api.js'

const EMPTY_HEADLINE = "Completely unique"
const EMPTY_SUBHEAD =
  "this track doesn't sound like anything in our reference catalog"

export default function SimilarityReport({
  caseA,
  neighbors,
  topPct,
  topLabel,
  topPercentile,
  topRawCosine,
  topSegment,
  querySpecificity,
  queryFile,
  contextToken,
}) {
  const top = neighbors?.[0]
  const top3 = (neighbors ?? []).slice(0, 3)
  // ADR-0004: headline no longer renders "Nth percentile match"; the
  // percentile field is kept in the wire shape for backward compat but
  // not surfaced to the user. Headline is `${label} · cosine ${cosine}`.
  const labelText = topLabel ? capitalizeLabel(topLabel) : null
  const showGenericNote = querySpecificity != null && querySpecificity < 0.50
  // One-row-open-at-a-time disclosure state. Click a row's chevron → that row
  // opens; clicking another closes the first.
  const [openIndex, setOpenIndex] = useState(null)

  // ---- Case A — match above threshold -------------------------------------
  if (caseA && top) {
    const topAudio = audioUrlFor(top.track)
    const topArt = artworkUrlFor(top.track, 300)
    // Headline rule per ADR-0004: calibrated label + raw cosine, no percentile.
    // Fall back to legacy raw-cosine percent only when the backend hasn't
    // shipped the calibrated label (older HF Space build).
    const useCalibrated = labelText != null
    return (
      <section>
        <Kicker>TOP MATCH</Kicker>
        <div className="mt-2 flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-4">
            <AudioPlayer src={topAudio} compact artwork={topArt} size={96} />
            <div>
              {useCalibrated ? (
                <>
                  <div className="flex flex-wrap items-baseline gap-3">
                    <span
                      className="font-display"
                      style={{
                        fontSize: '48px',
                        lineHeight: 0.95,
                        fontWeight: 700,
                        letterSpacing: '-0.03em',
                        color: 'var(--color-accent)',
                      }}
                    >
                      {labelText}
                    </span>
                    <span
                      className="font-mono text-[14px]"
                      style={{ color: 'var(--color-dim)' }}
                    >
                      · cosine {(topRawCosine ?? 0).toFixed(3)}
                    </span>
                  </div>
                  <div
                    className="mt-1.5 font-display"
                    style={{
                      fontSize: '28px',
                      fontWeight: 600,
                      letterSpacing: '-0.02em',
                    }}
                  >
                    {top.track?.title ?? top.trackId} —{' '}
                    <span style={{ color: 'var(--color-dim)' }}>
                      {top.track?.artist ?? 'Unknown artist'}
                    </span>
                  </div>
                  <div
                    className="mt-2 font-mono text-[11px]"
                    style={{ color: 'var(--color-faint)' }}
                  >
                    cosine {(topRawCosine ?? 0).toFixed(3)}
                    {topSegment != null && <> · segment {topSegment.toFixed(3)}</>}
                    {top.matchTimestamp && (
                      <>
                        {' '}
                        · {fmtTopMatchTs(top.matchTimestamp)}
                      </>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex flex-wrap items-baseline gap-3">
                    <span
                      className="font-display tabular-nums"
                      style={{
                        fontSize: '64px',
                        lineHeight: 0.9,
                        fontWeight: 700,
                        letterSpacing: '-0.04em',
                        color: 'var(--color-accent)',
                      }}
                    >
                      {topPct}%
                    </span>
                    <span className="text-base" style={{ color: 'var(--color-dim)' }}>
                      similar to
                    </span>
                  </div>
                  <div
                    className="mt-1.5 font-display"
                    style={{
                      fontSize: '32px',
                      fontWeight: 600,
                      letterSpacing: '-0.02em',
                    }}
                  >
                    {top.track?.title ?? top.trackId} —{' '}
                    <span style={{ color: 'var(--color-dim)' }}>
                      {top.track?.artist ?? 'Unknown artist'}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {top.track?.track_view_url && (
            <a
              href={top.track.track_view_url}
              target="_blank"
              rel="noopener noreferrer"
              className="whitespace-nowrap pt-2 font-mono text-[12px] no-underline"
              style={{ color: 'var(--color-dim)' }}
            >
              [↗ open in iTunes]
            </a>
          )}
        </div>

        {showGenericNote && (
          <div
            className="mt-4 rounded-sm px-3 py-2 font-mono text-[11px]"
            style={{
              background: 'var(--color-elev)',
              border: '1px solid var(--color-line)',
              color: 'var(--color-dim)',
            }}
          >
            Note: this generation pattern is broadly similar to many catalog
            tracks. The specific match is one of several close candidates.
          </div>
        )}

        <div className="mt-8">
          <Kicker>TOP 3 CLOSEST IN CATALOG</Kicker>
          <div className="mt-2">
            {top3.map((n, i) => (
              <SimilarityRow
                key={n.trackId ?? i}
                rank={i + 1}
                title={n.track?.title ?? n.trackId}
                artist={n.track?.artist ?? ''}
                similarity={n.meanPooledSimilarity}
                similarityLabel={n.similarityLabel}
                rawCosine={n.rawCosine ?? n.meanPooledSimilarity}
                linkOut={n.track?.track_view_url ?? n.track?.source_url}
                track={n.track}
                trackId={n.trackId}
                matchTimestamp={n.matchTimestamp}
                criteria={n.criteria}
                queryFile={queryFile}
                contextToken={contextToken}
                expanded={openIndex === i}
                onExpandToggle={() => setOpenIndex(openIndex === i ? null : i)}
              />
            ))}
          </div>
        </div>
      </section>
    )
  }

  // ---- Case B — completely unique -----------------------------------------
  return (
    <section>
      <div className="text-center">
        <Kicker className="mx-auto block">RESULT</Kicker>
        <div
          className="mt-2 font-display"
          style={{
            fontSize: '64px',
            lineHeight: 0.9,
            fontWeight: 700,
            letterSpacing: '-0.04em',
            color: 'var(--color-ink)',
          }}
        >
          {EMPTY_HEADLINE}
        </div>
        <div className="mt-2 text-base" style={{ color: 'var(--color-dim)' }}>
          {EMPTY_SUBHEAD}
        </div>
      </div>

      {top3.length > 0 && (
        <div className="mt-8">
          <Kicker>CLOSEST TRACKS · FOR REFERENCE, NOT MATCHES</Kicker>
          <div className="mt-2">
            {top3.map((n, i) => (
              <SimilarityRow
                key={n.trackId ?? i}
                rank={i + 1}
                title={n.track?.title ?? n.trackId}
                artist={n.track?.artist ?? ''}
                similarity={n.meanPooledSimilarity}
                similarityLabel={n.similarityLabel}
                rawCosine={n.rawCosine ?? n.meanPooledSimilarity}
                linkOut={n.track?.track_view_url ?? n.track?.source_url}
                track={n.track}
                trackId={n.trackId}
                matchTimestamp={n.matchTimestamp}
                criteria={n.criteria}
                queryFile={queryFile}
                contextToken={contextToken}
                expanded={openIndex === i + 100}
                onExpandToggle={() => setOpenIndex(openIndex === i + 100 ? null : i + 100)}
                isReference
              />
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function capitalizeLabel(label) {
  if (!label) return null
  return label.charAt(0).toUpperCase() + label.slice(1)
}

function fmtTopMatchTs(ts) {
  if (!ts) return null
  const fmt = (s) => {
    const n = Math.max(0, Math.floor(Number(s) || 0))
    const m = Math.floor(n / 60)
    const r = n % 60
    return `${m}:${r.toString().padStart(2, '0')}`
  }
  return `match ${fmt(ts.queryStartSec)}–${fmt(ts.queryEndSec)} ↔ ${fmt(ts.catalogStartSec)}–${fmt(ts.catalogEndSec)}`
}

function Kicker({ children, className = '' }) {
  return (
    <span
      className={`font-mono text-[12px] uppercase ${className}`}
      style={{
        color: 'var(--color-faint)',
        letterSpacing: '0.14em',
        fontWeight: 500,
      }}
    >
      {children}
    </span>
  )
}
