export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
export const clamp01 = (v) => clamp(v, 0, 1)

export function fmtDuration(sec) {
  if (sec == null || Number.isNaN(sec)) return '—'
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

// dB readouts use a typographic minus for a cleaner instrument look.
export function fmtDb(v, digits = 1) {
  const sign = v > 0 ? '+' : v < 0 ? '−' : '±'
  return `${sign}${Math.abs(v).toFixed(digits)}`
}
