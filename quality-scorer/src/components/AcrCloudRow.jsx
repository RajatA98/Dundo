/**
 * AcrCloudRow — one of the two independent ACRCloud signal rows on the ReportCard.
 *
 * Visual contract: ui_mockup_v2_suno_flare.html, `.acr-row` + `.acr-row.is-suno`.
 *
 * Two variants:
 *   variant="coverSongId"      → "ACRCLOUD · COVER SONG ID"        | "Cover match: ..." | "No cover match"
 *   variant="aiMusicDetector"  → "ACRCLOUD · AI MUSIC DETECTOR"    | "AI-generated (87%) — likely [SunoPill]"
 *
 * Independent rows — never composed into a single verdict (LOCKED_DECISIONS:
 * "ACRCloud — keep BOTH as separate signals"). Disagreement between Cover Song ID
 * and AI Music Detector is information, not a bug.
 *
 * Phase 5 wires this to the real ACRCloud response shape. Phase 3 scaffold
 * accepts a normalized props object so component-tests can exercise it without
 * a backend.
 *
 * @param {Object} props
 * @param {"coverSongId"|"aiMusicDetector"} props.variant
 * @param {Object|null} props.payload                   - normalized ACRCloud signal payload (see CODEX_PHASE_3)
 *                                                         null → not yet loaded
 * @param {"match"|"no_match"|"timeout"|"quota_exceeded"|"disabled"} [props.status]
 */
import SunoPill from './SunoPill.jsx'

const LABELS = {
  coverSongId: 'ACRCLOUD · COVER SONG ID',
  aiMusicDetector: 'ACRCLOUD · AI MUSIC DETECTOR',
}

export default function AcrCloudRow({ variant, payload, status = 'match' }) {
  const label = LABELS[variant] ?? 'ACRCLOUD'
  const isSuno =
    variant === 'aiMusicDetector' &&
    payload &&
    String(payload.likely_source || '').toLowerCase() === 'suno'

  return (
    <div
      className="flex flex-wrap items-baseline justify-between gap-2 border-t py-3"
      style={{ borderColor: 'var(--color-line)' }}
    >
      <span
        className="whitespace-nowrap font-mono text-[12px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.06em',
        }}
      >
        {label}
      </span>

      <span
        className="text-right text-sm"
        style={{ color: 'var(--color-ink)' }}
      >
        {/* TODO(codex): finish the per-variant rendering. The four payload
            shapes are documented in LOCKED_DECISIONS Q10. Suggested:
              - status === "disabled": muted "Signal unavailable in public demo"
              - status === "timeout" || "quota_exceeded": muted "service unavailable"
              - variant === "coverSongId" + status === "match":
                  "Cover match: <title> by <artist>, <score>%"
              - variant === "coverSongId" + status === "no_match":
                  muted "No cover match"
              - variant === "aiMusicDetector":
                  "AI-generated (<ai_probability>%) — " + (likely_source ? <SunoPill> if suno, else plain) */}
        {renderValue(variant, payload, status, isSuno)}
      </span>
    </div>
  )
}

function renderValue(variant, payload, status, isSuno) {
  if (status === 'disabled') {
    return (
      <span style={{ color: 'var(--color-dim)' }}>
        Signal unavailable in public demo — cached results visible on examples.
      </span>
    )
  }
  if (status === 'timeout' || status === 'quota_exceeded') {
    return (
      <span style={{ color: 'var(--color-dim)' }}>
        Second-opinion service unavailable.
      </span>
    )
  }
  if (variant === 'coverSongId') {
    if (status === 'no_match' || !payload || !payload.title) {
      return <span style={{ color: 'var(--color-dim)' }}>No cover match</span>
    }
    return (
      <>
        Cover match: &ldquo;{payload.title}&rdquo; by {payload.artist}, {payload.score}%
      </>
    )
  }
  if (variant === 'aiMusicDetector') {
    if (!payload) {
      return <span style={{ color: 'var(--color-dim)' }}>—</span>
    }
    // Per LOCKED_DECISIONS Q10, ai_probability is 0–100. Just round + clamp.
    const aiPct = Math.max(0, Math.min(100, Math.round(payload.ai_probability ?? 0)))

    // Dispatch on verdict — "AI-generated", "human", and "no_vocals" are three
    // distinct states, not three flavors of the same sentence.
    if (payload.verdict === 'human') {
      return (
        <>
          Likely human{' '}
          <span style={{ color: 'var(--color-dim)' }}>
            ({aiPct}% AI probability)
          </span>
        </>
      )
    }
    if (payload.verdict === 'no_vocals') {
      return (
        <span style={{ color: 'var(--color-dim)' }}>
          No vocals detected
        </span>
      )
    }
    // Default — ai_generated.
    return (
      <>
        AI-generated ({aiPct}%) —{' '}
        {isSuno
          ? <SunoPill />
          : payload.likely_source
            ? <span>likely {payload.likely_source}</span>
            : <span style={{ color: 'var(--color-dim)' }}>source unclear</span>}
      </>
    )
  }
  return null
}
