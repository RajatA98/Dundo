import { useState } from 'react'
import AudioPlayer from './AudioPlayer.jsx'
import SectionComparePanel from './SectionComparePanel.jsx'
import { audioUrlFor, artworkUrlFor } from '../lib/api.js'

/**
 * SimilarityRow — one row in the top-3 closest tracks list.
 *
 * Per ADR-0004:
 *   - the percentile column is gone; the right edge shows just the
 *     similarityLabel ("very close" / "close" / etc.) with the raw cosine
 *     in small monospace beneath.
 *   - a disclosure chevron at the left expands SectionComparePanel —
 *     side-by-side snippet players + per-criterion comparison table.
 *
 * The bar driven by raw cosine stays; Codex explicitly endorsed it as a
 * visual spread device.
 */
export default function SimilarityRow({
  rank,
  title,
  artist,
  similarity,
  similarityLabel,
  rawCosine,
  matchTimestamp,
  criteria,
  linkOut,
  track,
  trackId,
  queryFile,
  contextToken,
  expanded = false,
  onExpandToggle,
  isReference = false,
}) {
  // Local-state fallback so the row still expands when no parent coordinator
  // is wired (e.g., in tests). When SimilarityReport passes expanded + onExpandToggle,
  // those win and let one-open-at-a-time work.
  const [localExpanded, setLocalExpanded] = useState(false)
  const isOpen = onExpandToggle ? expanded : localExpanded
  const handleToggle = () => {
    if (onExpandToggle) onExpandToggle()
    else setLocalExpanded((v) => !v)
  }

  const audioUrl = audioUrlFor(track)
  const artworkUrl = artworkUrlFor(track, 100)
  const cosine = Number(rawCosine ?? similarity) || 0
  // Bar width = raw cosine (the actual signal strength). Codex § "Sanity checks":
  // bar width using raw cosine is OK as a visual spread device.
  const barPct = Math.max(2, Math.min(100, Math.round(cosine * 100)))
  const widthStyle = { width: `${barPct}%` }
  const cosineForDetail = cosine.toFixed(3)
  const labelText = capitalize(similarityLabel)

  return (
    <div
      className="border-t"
      style={{
        borderColor: 'var(--color-line)',
        opacity: isReference ? 0.6 : 1,
      }}
    >
      <div
        className="grid items-center gap-3 py-3"
        style={{
          gridTemplateColumns: '24px 18px 40px minmax(160px, 1.4fr) 2fr 130px',
        }}
      >
        <button
          type="button"
          onClick={handleToggle}
          aria-label={isOpen ? 'Collapse details' : 'Expand details'}
          aria-expanded={isOpen}
          className="grid h-5 w-5 place-items-center rounded-sm transition-transform hover:bg-line"
          style={{
            transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
            color: 'var(--color-faint)',
          }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <path d="M3 1.5l4 3.5-4 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>

        <span
          className="font-mono text-xs"
          style={{ color: 'var(--color-faint)' }}
        >
          {rank}
        </span>

        <AudioPlayer src={audioUrl} compact artwork={artworkUrl} size={40} />

        <span className="flex flex-col">
          <span className="text-[15px] leading-snug">
            {linkOut ? (
              <a
                href={linkOut}
                target="_blank"
                rel="noopener noreferrer"
                className="no-underline hover:underline"
                style={{ color: 'inherit' }}
              >
                {title}
              </a>
            ) : (
              title
            )}{' '}
            <span style={{ color: 'var(--color-dim)' }}>— {artist}</span>
          </span>
        </span>

        <span
          className="h-1 rounded-sm"
          style={{ background: 'var(--color-line)' }}
        >
          <span
            className="block h-full rounded-sm"
            style={{
              ...widthStyle,
              background: isReference ? '#9AA0A8' : 'var(--color-accent)',
            }}
          />
        </span>

        <span
          className="text-right text-[12px]"
          style={{ color: isReference ? 'var(--color-dim)' : 'var(--color-ink)' }}
          title={`raw cosine ${cosineForDetail}`}
        >
          <span className="block">{labelText}</span>
          <span
            className="block font-mono text-[10px] tabular-nums"
            style={{ color: 'var(--color-faint)' }}
          >
            cos {cosineForDetail}
          </span>
        </span>
      </div>

      {isOpen && (
        <SectionComparePanel
          matchTimestamp={matchTimestamp}
          criteria={criteria}
          queryFile={queryFile}
          catalogTrack={track}
          trackId={trackId}
          contextToken={contextToken}
        />
      )}
    </div>
  )
}


function capitalize(label) {
  if (!label || typeof label !== 'string') return 'Match'
  return label.charAt(0).toUpperCase() + label.slice(1)
}
