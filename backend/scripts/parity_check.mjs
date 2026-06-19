// Runs the REAL JS `computeReport` over a fixed set of raw inputs and emits
// {name, raw, expected} JSON on stdout. The Python parity test (pytest) shells
// out to this and asserts the Python port produces identical reports.
//
// The JS files are ESM (quality-scorer/package.json has type: "module"); this
// script is .mjs so node treats it as ESM regardless of nearby package.json.

import { computeReport } from '../../quality-scorer/src/lib/scoring.js'

const CASES = [
  { name: 'clean',                 raw: { silence: 2.1,  clipping: 0.04, noise: 0.18, truncation: -34, duration: 192, channel: 1.2,  dynamics: 11.4 } },
  { name: 'clipping_fail',         raw: { silence: 1.4,  clipping: 4.6,  noise: 0.31, truncation: -25, duration: 148, channel: 2.0,  dynamics: 5.2  } },
  { name: 'silence_fail',          raw: { silence: 78,   clipping: 0,    noise: 0.22, truncation: -38, duration: 211, channel: 0.6,  dynamics: 3.1  } },
  { name: 'noise_fail',            raw: { silence: 0.9,  clipping: 0.8,  noise: 0.71, truncation: -19, duration: 176, channel: 3.4,  dynamics: 6.1  } },
  { name: 'truncation_fail',       raw: { silence: 3.2,  clipping: 0.06, noise: 0.20, truncation: -2.4,duration: 38,  channel: 1.1,  dynamics: 9.7  } },
  { name: 'dead_channel',          raw: { silence: 1.8,  clipping: 0.05, noise: 0.25, truncation: -28, duration: 175, channel: 26,   dynamics: 9.5  } },
  { name: 'duration_short',        raw: { silence: 2.2,  clipping: 0.02, noise: 0.20, truncation: -30, duration: 8,   channel: 1.5,  dynamics: 10.1 } },
  { name: 'duration_long',         raw: { silence: 1.4,  clipping: 0.03, noise: 0.22, truncation: -32, duration: 460, channel: 1.7,  dynamics: 9.9  } },
  { name: 'dynamics_fail_only',    raw: { silence: 2.0,  clipping: 0.05, noise: 0.21, truncation: -29, duration: 180, channel: 1.8,  dynamics: 3.0  } },
  { name: 'all_warns',             raw: { silence: 16,   clipping: 0.5,  noise: 0.45, truncation: -10, duration: 180, channel: 12,   dynamics: 6    } },
  { name: 'multi_critical_fail',   raw: { silence: 50,   clipping: 3.5,  noise: 0.60, truncation: -3,  duration: 7,   channel: 22,   dynamics: 2    } },
  { name: 'borderline_keep',       raw: { silence: 14.9, clipping: 0.29, noise: 0.39, truncation: -19, duration: 21,  channel: 8.9,  dynamics: 7.1  } },
  { name: 'just_warn_silence',     raw: { silence: 25,   clipping: 0.1,  noise: 0.25, truncation: -28, duration: 180, channel: 2,    dynamics: 9    } },
  { name: 'just_warn_clip',        raw: { silence: 2,    clipping: 1.2,  noise: 0.25, truncation: -28, duration: 180, channel: 2,    dynamics: 9    } },
  { name: 'high_noise_only',       raw: { silence: 1,    clipping: 0.05, noise: 0.85, truncation: -28, duration: 180, channel: 2,    dynamics: 9    } },
  { name: 'dynamics_warn_only',    raw: { silence: 1.5,  clipping: 0.05, noise: 0.18, truncation: -31, duration: 180, channel: 1.5,  dynamics: 5    } },
  { name: 'edge_truncation_warn',  raw: { silence: 2,    clipping: 0.05, noise: 0.21, truncation: -10, duration: 180, channel: 2,    dynamics: 9    } },
  { name: 'short_track_warn_dur',  raw: { silence: 2,    clipping: 0.05, noise: 0.21, truncation: -28, duration: 18,  channel: 2,    dynamics: 9    } },
  { name: 'channel_warn',          raw: { silence: 2,    clipping: 0.05, noise: 0.21, truncation: -28, duration: 180, channel: 11,   dynamics: 9    } },
  { name: 'mid_score',             raw: { silence: 5,    clipping: 0.2,  noise: 0.42, truncation: -15, duration: 100, channel: 5,    dynamics: 8    } },
]

const out = CASES.map((c) => ({
  name: c.name,
  raw: c.raw,
  expected: computeReport(c.raw),
}))

process.stdout.write(JSON.stringify(out, null, 2))
