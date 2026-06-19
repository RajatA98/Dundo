import { mulberry32, pick, range, chance } from './prng.js'
import { computeReport } from './scoring.js'
import { clamp01 } from './format.js'

const WAVE_N = 180

// ----- waveform synthesis (mock peak envelope, 0..1) --------------------------

function baseEnvelope(rng) {
  const out = new Array(WAVE_N)
  const sections = 3 + Math.floor(rng() * 3)
  const secLen = WAVE_N / sections
  const phase = rng() * Math.PI * 2
  const wob = range(rng, 1.5, 3.5)
  for (let i = 0; i < WAVE_N; i++) {
    const t = i / WAVE_N
    const macro = 0.35 + 0.45 * Math.sin(t * Math.PI)
    const section = 0.5 + 0.5 * Math.sin((i / secLen) * Math.PI)
    const groove = 0.5 + 0.5 * Math.sin(t * Math.PI * 2 * wob + phase)
    let v = macro * 0.7 + section * 0.12 + groove * 0.12 + (rng() - 0.5) * 0.18
    if (i % 8 === 0) v += 0.08
    out[i] = clamp01(v * range(rng, 0.92, 1))
  }
  return out
}

function applyKind(peaks, rng, kind) {
  const N = peaks.length
  const p = peaks.slice()
  const problems = []
  if (kind === 'clip') {
    const regions = 2 + Math.floor(rng() * 2)
    for (let r = 0; r < regions; r++) {
      const len = 6 + Math.floor(rng() * 10)
      const from = Math.floor(rng() * (N - len))
      for (let i = from; i < from + len; i++) p[i] = range(rng, 0.97, 1)
      problems.push({ type: 'clip', from, to: from + len })
    }
  } else if (kind === 'silence') {
    if (chance(rng, 0.5)) {
      for (let i = 0; i < N; i++) p[i] = range(rng, 0.005, 0.04)
      problems.push({ type: 'silence', from: 0, to: N })
    } else {
      const len = Math.floor(N * range(rng, 0.3, 0.55))
      const from = Math.floor(rng() * (N - len))
      for (let i = from; i < from + len; i++) p[i] = range(rng, 0.005, 0.035)
      problems.push({ type: 'silence', from, to: from + len })
    }
  } else if (kind === 'truncation') {
    const cut = Math.floor(N * range(rng, 0.82, 0.94))
    for (let i = Math.max(0, cut - 4); i < cut; i++) p[i] = range(rng, 0.7, 0.95)
    for (let i = cut; i < N; i++) p[i] = 0
    problems.push({ type: 'truncation', from: cut, to: N })
  } else if (kind === 'noise') {
    for (let i = 0; i < N; i++) p[i] = range(rng, 0.5, 0.95)
  }
  return { peaks: p, problems }
}

// ----- raw signal value synthesis --------------------------------------------

function goodRaw(rng, durationSec) {
  return {
    silence: range(rng, 0.5, 9),
    clipping: range(rng, 0, 0.22),
    noise: range(rng, 0.1, 0.34),
    truncation: range(rng, -40, -22),
    duration: durationSec,
    channel: range(rng, 0.3, 6),
    dynamics: range(rng, 7.5, 13),
  }
}

function injectFailure(raw, rng, mode) {
  const r = { ...raw }
  switch (mode) {
    case 'clipping':
      r.clipping = range(rng, 3, 9)
      r.dynamics = range(rng, 4.5, 7)
      break
    case 'silence':
      r.silence = range(rng, 45, 92)
      r.dynamics = range(rng, 2, 6)
      break
    case 'noise':
      r.noise = range(rng, 0.58, 0.85)
      r.clipping = range(rng, 0.1, 1.4)
      break
    case 'truncation':
      r.truncation = range(rng, -5, -1)
      break
    case 'duration':
      r.duration = chance(rng, 0.5) ? range(rng, 4, 11) : range(rng, 430, 520)
      break
    case 'channel':
      r.channel = range(rng, 20, 30)
      break
    case 'dynamics':
      r.dynamics = range(rng, 2, 3.8)
      break
    default:
      break
  }
  return r
}

const WAVE_KIND = {
  clipping: 'clip',
  silence: 'silence',
  truncation: 'truncation',
  noise: 'noise',
}

function buildTrack(meta, rng) {
  const report = computeReport(meta.raw)
  const kind = meta.kind ?? WAVE_KIND[report.primaryFail] ?? 'clean'
  const wf = applyKind(baseEnvelope(rng), rng, kind)
  return {
    id: meta.id,
    title: meta.title,
    genre: meta.genre,
    durationSec: meta.raw.duration,
    source: meta.source ?? 'corpus',
    example: meta.example,
    chipLabel: meta.chipLabel,
    waveform: wf.peaks,
    problems: wf.problems,
    ...report,
  }
}

// ----- curated examples (the Scorer page chips) ------------------------------

const EXAMPLE_DEFS = [
  {
    id: 'ex-clean',
    example: 'clean',
    chipLabel: 'Clean',
    title: 'Neon Tide Pools',
    genre: 'Synthwave',
    kind: 'clean',
    raw: { silence: 2.1, clipping: 0.04, noise: 0.18, truncation: -34, duration: 192, channel: 1.2, dynamics: 11.4 },
  },
  {
    id: 'ex-clipped',
    example: 'clipped',
    chipLabel: 'Clipped',
    title: 'Crush the Citadel',
    genre: 'Trap Metal',
    kind: 'clip',
    raw: { silence: 1.4, clipping: 4.6, noise: 0.31, truncation: -25, duration: 148, channel: 2.0, dynamics: 5.2 },
  },
  {
    id: 'ex-silent',
    example: 'silent',
    chipLabel: 'Silent',
    title: 'Untitled Render 0447',
    genre: 'Ambient',
    kind: 'silence',
    raw: { silence: 78, clipping: 0, noise: 0.22, truncation: -38, duration: 211, channel: 0.6, dynamics: 3.1 },
  },
  {
    id: 'ex-noisy',
    example: 'noisy',
    chipLabel: 'Noisy',
    title: 'Static Cathedral',
    genre: 'Industrial',
    kind: 'noise',
    raw: { silence: 0.9, clipping: 0.8, noise: 0.71, truncation: -19, duration: 176, channel: 3.4, dynamics: 6.1 },
  },
  {
    id: 'ex-truncated',
    example: 'truncated',
    chipLabel: 'Truncated',
    title: 'Half-Light Sermon',
    genre: 'Indie Folk',
    kind: 'truncation',
    raw: { silence: 3.2, clipping: 0.06, noise: 0.2, truncation: -2.4, duration: 38, channel: 1.1, dynamics: 9.7 },
  },
]

export const EXAMPLES = EXAMPLE_DEFS.map((m, i) =>
  buildTrack({ ...m, source: 'example' }, mulberry32(101 + i * 7)),
)

// ----- the ~300-track corpus --------------------------------------------------

const GENRES = [
  'Synthwave', 'Lo-fi Hip-Hop', 'Ambient', 'Trap', 'Indie Folk', 'House',
  'Drum & Bass', 'Cinematic', 'Industrial', 'Jazz Fusion', 'Phonk',
  'Orchestral', 'Hyperpop', 'Dream Pop', 'Techno',
]

const ADJ = [
  'Neon', 'Velvet', 'Glass', 'Midnight', 'Paper', 'Hollow', 'Golden', 'Crimson',
  'Static', 'Phantom', 'Marble', 'Electric', 'Quiet', 'Brittle', 'Distant',
  'Molten', 'Saltwater', 'Lunar', 'Concrete', 'Wax', 'Cobalt', 'Feral', 'Soft',
]
const NOUN = [
  'Tide', 'Cathedral', 'Engine', 'Garden', 'Mirror', 'Harbor', 'Circuit', 'Sermon',
  'Avenue', 'Bloom', 'Furnace', 'Anthem', 'Lantern', 'Drift', 'Pulse', 'Meridian',
  'Hymn', 'Corridor', 'Ode', 'Lattice', 'Ravine', 'Signal',
]

// Failure-mode mix among the dropped slice (roughly realistic for a free-tier dump).
const FAIL_MODES = [
  'silence', 'silence', 'clipping', 'clipping', 'clipping', 'noise', 'noise',
  'truncation', 'truncation', 'channel', 'duration', 'dynamics',
]

function makeTitle(rng) {
  if (chance(rng, 0.08)) return `Untitled Render ${String(Math.floor(rng() * 9999)).padStart(4, '0')}`
  if (chance(rng, 0.12)) return `${pick(rng, NOUN)} of ${pick(rng, NOUN)}`
  return `${pick(rng, ADJ)} ${pick(rng, NOUN)}`
}

function generateCorpus(n) {
  const rng = mulberry32(0x50c1)
  const out = []
  for (let i = 0; i < n; i++) {
    const durationSec = chance(rng, 0.08) ? range(rng, 60, 95) : range(rng, 95, 305)
    let raw = goodRaw(rng, Math.round(durationSec))
    if (chance(rng, 0.18)) raw = injectFailure(raw, rng, pick(rng, FAIL_MODES))
    out.push(
      buildTrack(
        {
          id: `trk-${String(i + 1).padStart(3, '0')}`,
          title: makeTitle(rng),
          genre: pick(rng, GENRES),
          raw,
          source: 'corpus',
        },
        rng,
      ),
    )
  }
  return out
}

export const CORPUS = generateCorpus(300)
export const ALL_GENRES = GENRES

export function corpusStats(tracks = CORPUS) {
  const total = tracks.length
  const dropped = tracks.filter((t) => t.verdict === 'drop')
  const byMode = {}
  for (const t of dropped) {
    const m = t.primaryFail ?? 'other'
    byMode[m] = (byMode[m] ?? 0) + 1
  }
  return {
    total,
    kept: total - dropped.length,
    dropped: dropped.length,
    keepPct: total ? Math.round(((total - dropped.length) / total) * 100) : 0,
    byMode,
  }
}

export function getTrackById(id) {
  return EXAMPLES.find((t) => t.id === id) ?? CORPUS.find((t) => t.id === id) ?? null
}

// ----- mock "analyze an upload" ----------------------------------------------

function hashStr(s) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

// Deterministic from filename so re-uploading the same file is stable.
export function analyzeUpload(filename) {
  const rng = mulberry32(hashStr(filename || 'upload') || 1)
  const durationSec = Math.round(range(rng, 80, 280))
  let raw = goodRaw(rng, durationSec)
  if (chance(rng, 0.4)) raw = injectFailure(raw, rng, pick(rng, FAIL_MODES))
  return buildTrack(
    {
      id: 'upload',
      title: filename ? filename.replace(/\.[^.]+$/, '') : 'Uploaded track',
      genre: 'Uploaded',
      raw,
      source: 'upload',
    },
    rng,
  )
}
