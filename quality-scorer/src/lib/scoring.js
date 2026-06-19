import { SIGNALS, evaluateSignal } from './signals.js'
import { clamp } from './format.js'

// Critical signals are weighted hardest; a broken track passing is the costly error.
const WEIGHT = {
  silence: 1.0,
  clipping: 1.0,
  noise: 0.9,
  truncation: 0.85,
  channel: 0.7,
  duration: 0.6,
  dynamics: 0.4,
}

const FAIL_PHRASE = {
  silence: 'mostly dead air',
  clipping: 'hard clipping',
  noise: 'noise-dominated spectrum',
  truncation: 'cut off mid-phrase',
  duration: 'length out of range',
  channel: 'collapsed stereo channel',
  dynamics: 'no dynamic range',
}

// Turns raw measured values into a full report. The verdict is precision-first:
// any CRITICAL signal failing → DROP. Dynamics can fail without forcing a drop.
export function computeReport(raw) {
  const signals = SIGNALS.map((s) => {
    const ev = evaluateSignal(s.id, raw[s.id])
    return {
      id: s.id,
      label: s.label,
      short: s.short,
      critical: s.critical,
      threshold: s.threshold,
      blurb: s.blurb,
      value: raw[s.id],
      ...ev,
    }
  })

  let penalty = 0
  for (const s of signals) penalty += Math.pow(s.severity, 1.4) * (WEIGHT[s.id] ?? 0.5) * 27
  const score = Math.round(clamp(100 - penalty, 0, 100))

  const failed = signals.filter((s) => s.status === 'fail')
  const criticalFails = failed.filter((s) => s.critical).sort((a, b) => b.severity - a.severity)
  const verdict = criticalFails.length > 0 ? 'drop' : 'keep'
  const primaryFail = verdict === 'drop' ? criticalFails[0].id : null

  let reason
  if (verdict === 'drop') {
    const w = criticalFails[0]
    const extra = criticalFails.length > 1 ? ` · +${criticalFails.length - 1} more` : ''
    reason = `Dropped — ${FAIL_PHRASE[w.id]} (${w.label.toLowerCase()} ${w.display})${extra}.`
  } else {
    const warns = signals.filter((s) => s.status === 'warn')
    reason = warns.length
      ? `Kept — within bounds; ${warns.length} signal${warns.length > 1 ? 's' : ''} flagged for review.`
      : 'Kept — all technical signals within bounds.'
  }

  return { score, verdict, primaryFail, reason, signals, failModes: failed.map((s) => s.id) }
}
