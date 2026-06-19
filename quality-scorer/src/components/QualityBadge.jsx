/**
 * QualityBadge — inline badge + click-to-expand 7-signal breakdown.
 *
 * Phase 3 scaffold. Visual contract: ui_mockup_v2_suno_flare.html `.quality`
 * + `.quality-breakdown` blocks.
 *
 * The inherited 7-signal quality detector is demoted to a secondary status
 * badge per LOCKED_DECISIONS "Quality status badge (secondary)".
 * The badge renders one of two states:
 *
 *   - "ok"     → small green dot + "Track quality: ok"
 *   - "issues" → small amber dot + "Track quality: issues detected"
 *
 * The breakdown shows the existing 7 signals from /analyze:
 *   CLIPPING, SILENCE (lead/tail), LOUDNESS, CHANNELS, SAMPLE RATE, DURATION, DC OFFSET.
 *
 * The badge consumes the existing /analyze response shape, decoupled from
 * /neighbors so the quality signal stays orthogonal to the discovery flow.
 *
 * @param {Object} props
 * @param {Object|null} props.analyze - the legacy /analyze response (Track-shape JSON).
 *                                       null → still loading / not yet called.
 */
import { useState } from 'react'

export default function QualityBadge({ analyze }) {
  const [open, setOpen] = useState(false)

  if (!analyze) {
    return (
      <span
        className="inline-flex items-center gap-2 text-sm"
        style={{ color: 'var(--color-dim)' }}
      >
        <span
          className="block h-2 w-2 rounded-full"
          style={{ background: 'var(--color-line)' }}
        />
        Track quality: checking…
      </span>
    )
  }

  // TODO(codex): the /analyze response carries a `problems` array and a
  // `signals` per-signal breakdown. Decide ok vs issues by:
  //   - ok    when problems.length === 0
  //   - issues otherwise
  // See backend/backend/scoring.py for the live shape if uncertain.
  const hasIssues = Array.isArray(analyze.problems) && analyze.problems.length > 0

  return (
    <div>
      <span
        className="inline-flex items-center gap-2 text-sm"
        style={{ color: 'var(--color-ink)' }}
      >
        <span
          className="block h-2 w-2 rounded-full"
          style={{ background: hasIssues ? '#D98A2B' : 'var(--color-accent)' }}
        />
        Track quality: {hasIssues ? 'issues detected' : 'ok'}
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="cursor-pointer border-b font-mono text-[12px]"
          style={{
            color: 'var(--color-dim)',
            borderColor: 'var(--color-line)',
            background: 'transparent',
          }}
        >
          {open ? '(hide breakdown)' : '(see breakdown)'}
        </button>
      </span>

      {open && (
        <div
          className="mt-2 rounded-sm border"
          style={{ borderColor: 'var(--color-line)' }}
        >
          {/* TODO(codex): map analyze.signals into rows. Each row is:
                <div class="qsignal">
                  <span class="qname">CLIPPING</span>
                  <span class="qval ok">none detected</span>
                </div>
              with `qval.ok` (green) / `qval.flag` (amber) class chosen by
              the per-signal status in the /analyze response.
              See ui_mockup_v2_suno_flare.html lines 718–727 for the literal markup. */}
          <div className="px-3 py-2 font-mono text-[12px]" style={{ color: 'var(--color-dim)' }}>
            {/* placeholder — Codex replaces */}
            7-signal breakdown placeholder.
          </div>
        </div>
      )}
    </div>
  )
}
