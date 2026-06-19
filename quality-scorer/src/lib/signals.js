import { clamp01, fmtDuration, fmtDb } from './format.js'

// The seven heuristic signals. `critical` signals can force a DROP on their own
// (precision-first: a broken track reaching the feed is highly visible).
export const SIGNALS = [
  {
    id: 'silence',
    label: 'Silence ratio',
    short: 'SIL',
    critical: true,
    threshold: 'fail > 35%  ·  warn > 15%',
    blurb:
      'Share of frames below the noise floor. High values mean dead air or a render that never produced sound.',
  },
  {
    id: 'clipping',
    label: 'Clipping',
    short: 'CLIP',
    critical: true,
    threshold: 'fail > 2%  ·  warn > 0.3%',
    blurb:
      'Share of samples pinned at full scale. Indicates hard digital distortion baked into the audio.',
  },
  {
    id: 'noise',
    label: 'Noise · spectral flatness',
    short: 'NOISE',
    critical: true,
    threshold: 'fail > 0.55  ·  warn > 0.40',
    blurb:
      'Flatness of the spectrum, 0 (tonal) → 1 (noise-like). High flatness reads as hiss, static, or a failed generation.',
  },
  {
    id: 'truncation',
    label: 'Edge truncation',
    short: 'TRUNC',
    critical: true,
    threshold: 'fail > −6 dB at an edge',
    blurb:
      'Energy at the very first/last frames. A loud edge means the track was cut mid-phrase rather than ending.',
  },
  {
    id: 'duration',
    label: 'Duration',
    short: 'DUR',
    critical: true,
    threshold: 'keep 20s – 360s',
    blurb: 'Total length. Stubs and runaway renders fall outside the expected band.',
  },
  {
    id: 'channel',
    label: 'Channel balance',
    short: 'CHAN',
    critical: true,
    threshold: 'fail > 18 dB  ·  warn > 9 dB',
    blurb:
      'Level difference between left and right. A large gap means a dead or collapsed stereo channel.',
  },
  {
    id: 'dynamics',
    label: 'Dynamic range',
    short: 'DYN',
    critical: false,
    threshold: 'warn < 7 dB  ·  flag < 4 dB',
    blurb:
      'Crest factor — peak vs RMS level. Very low range reads as over-compressed or lifeless, but rarely broken on its own.',
  },
]

export const SIGNAL_BY_ID = Object.fromEntries(SIGNALS.map((s) => [s.id, s]))

// Maps a raw measured value → { status, severity (0 best → 1 worst), display }.
// A real backend supplies the raw value; this evaluator owns the verdict logic.
export function evaluateSignal(id, value) {
  switch (id) {
    case 'silence': {
      const status = value > 35 ? 'fail' : value > 15 ? 'warn' : 'pass'
      return { status, severity: clamp01(value / 60), display: `${value.toFixed(1)}%` }
    }
    case 'clipping': {
      const status = value > 2 ? 'fail' : value > 0.3 ? 'warn' : 'pass'
      return { status, severity: clamp01(value / 6), display: `${value.toFixed(2)}%` }
    }
    case 'noise': {
      const status = value > 0.55 ? 'fail' : value > 0.4 ? 'warn' : 'pass'
      return { status, severity: clamp01((value - 0.1) / 0.7), display: value.toFixed(2) }
    }
    case 'truncation': {
      // value is dB at the loudest edge; closer to 0 = harder cut = worse.
      const status = value > -6 ? 'fail' : value > -18 ? 'warn' : 'pass'
      return { status, severity: clamp01((value + 42) / 42), display: `${fmtDb(value)} dB` }
    }
    case 'duration': {
      const status = value < 12 || value > 420 ? 'fail' : value < 20 || value > 360 ? 'warn' : 'pass'
      const ideal = clamp01(Math.min(Math.abs(value - 20), Math.abs(value - 360)) / 60)
      const sev = status === 'pass' ? 0 : status === 'warn' ? 0.4 + ideal * 0.2 : 0.85
      return { status, severity: sev, display: fmtDuration(value) }
    }
    case 'channel': {
      const status = value > 18 ? 'fail' : value > 9 ? 'warn' : 'pass'
      return { status, severity: clamp01(value / 30), display: `${value.toFixed(1)} dB` }
    }
    case 'dynamics': {
      const status = value < 4 ? 'fail' : value < 7 ? 'warn' : 'pass'
      return { status, severity: clamp01((12 - value) / 12), display: `${value.toFixed(1)} dB` }
    }
    default:
      return { status: 'pass', severity: 0, display: String(value) }
  }
}

export const STATUS_LABEL = { pass: 'PASS', warn: 'WARN', fail: 'FAIL' }
