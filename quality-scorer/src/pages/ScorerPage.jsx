import { useState } from 'react'
import Hero from '../components/Hero.jsx'
import DropZone from '../components/DropZone.jsx'
import ArtistResults, { EmptyState } from '../components/ArtistResults.jsx'
import { neighborsUpload } from '../lib/api.js'

/**
 * Landing page — drop an AI track, meet the top-3 indie artists who sound
 * like it. Realizes the approved Dundo.dc.html app view: hero → drop zone →
 * artist results (Case A) → the honest Case-B state.
 *
 * Phase 3 wired: the drop zone POSTs the upload to `/neighbors` and renders the
 * real artist-framed response (`ArtistNeighborsResponse` — top-3, threshold-gated,
 * never padded). `contextToken` is threaded to the cards so each "why this
 * resonates" can hydrate lazily via `/narrative`.
 */
export default function ScorerPage() {
  const [phase, setPhase] = useState('idle') // 'idle' | 'analyzing' | 'results' | 'empty' | 'error'
  const [matches, setMatches] = useState([])
  const [contextToken, setContextToken] = useState(null)
  const [querySummary, setQuerySummary] = useState(null)
  const [error, setError] = useState('')
  const [queryUrl, setQueryUrl] = useState(null)

  const onFile = async (file) => {
    if (!file) return
    const ok = file.type.startsWith('audio/') || /\.(mp3|wav|flac|ogg|m4a)$/i.test(file.name)
    if (!ok) {
      setError(`Couldn't read "${file.name}" — expected an audio file (mp3, wav, flac, ogg, m4a).`)
      setPhase('error')
      return
    }
    setError('')
    setQueryUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(file)
    })
    setPhase('analyzing')
    try {
      const res = await neighborsUpload(file, 3)
      const m = Array.isArray(res?.matches) ? res.matches : []
      setMatches(m)
      setContextToken(res?.contextToken || null)
      setQuerySummary(res?.querySummary || null)
      setPhase(m.length > 0 ? 'results' : 'empty')
    } catch (e) {
      setError(e?.message || 'Something went wrong analyzing your track — please try again.')
      setPhase('error')
    }
  }

  return (
    <>
      <Hero />
      <DropZone onFile={onFile} disabled={phase === 'analyzing'} />

      {phase === 'analyzing' && <Analyzing />}
      {phase === 'error' && <ErrorNote msg={error} />}
      {(phase === 'results' || phase === 'empty') && queryUrl && <YourTrack url={queryUrl} stats={querySummary} />}
      {phase === 'results' && <ArtistResults artists={matches} contextToken={contextToken} queryUrl={queryUrl} />}
      {phase === 'empty' && <EmptyState />}
    </>
  )
}

function YourTrack({ url, stats }) {
  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '56px 28px 0' }}>
      <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 14 }}>
        Your track
      </div>
      <div style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 16, padding: '16px 18px' }}>
        <audio src={url} controls style={{ width: '100%', height: 38 }} />
        {stats && <SongStats stats={stats} />}
      </div>
    </section>
  )
}

const fmtDuration = (sec) => {
  if (sec == null) return null
  const s = Math.round(sec)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s)
const keyConfLabel = (c) => (c == null ? null : c >= 0.7 ? 'high' : c >= 0.5 ? 'moderate' : 'low')
const ENERGY_FILL = { Low: 2, Medium: 3, High: 4 }

function Stat({ label, value, sub }) {
  if (value == null || value === '') return null
  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--color-faint)', marginBottom: 5 }}>{label}</div>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 19, fontWeight: 500, color: 'var(--color-ink)', lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--color-faint)', marginTop: 3 }}>{sub}</div>}
    </div>
  )
}

/** "Your song's stats" — honest snapshot. Hard numbers from the audio (tempo, key,
 *  length, energy band) + real propagated tags (mood, genre). No fake decimals. */
function SongStats({ stats }) {
  const tags = stats.tags || {}
  const moods = (tags.mood || []).slice(0, 2).map((t) => cap(t.label)).join(', ')
  const genres = (tags.genre || []).slice(0, 2).map((t) => cap(t.label)).join(', ')
  const conf = keyConfLabel(stats.keyConfidence)
  const energyFill = ENERGY_FILL[stats.energyBand] || 0

  return (
    <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--color-line)' }}>
      <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 14 }}>
        Your song&rsquo;s stats <span style={{ color: 'var(--color-faint)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>· measured from the audio</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 20 }}>
        <Stat label="Tempo" value={stats.tempoBpm ? `≈ ${stats.tempoBpm} BPM` : null} />
        <Stat
          label="Key"
          value={stats.key ? `${stats.key} ${stats.mode || ''}`.trim() : null}
          sub={conf ? `${conf} confidence` : null}
        />
        <Stat label="Length" value={fmtDuration(stats.durationSec)} />
        {stats.energyBand && (
          <Stat
            label="Energy"
            value={
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <span style={{ display: 'inline-flex', gap: 3 }}>
                  {[0, 1, 2, 3, 4].map((i) => (
                    <span key={i} style={{ width: 7, height: 7, borderRadius: 99, background: i < energyFill ? 'var(--color-teal)' : 'var(--color-line)' }} />
                  ))}
                </span>
                <span style={{ fontSize: 15 }}>{stats.energyBand}</span>
              </span>
            }
          />
        )}
        <Stat label="Mood" value={moods || null} sub={moods ? 'closest tags' : null} />
        <Stat label="Genre" value={genres || null} sub={genres ? 'closest tags' : null} />
      </div>
    </div>
  )
}

function Analyzing() {
  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '56px 28px 0' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          background: 'var(--color-paper)',
          border: '1px solid var(--color-line)',
          borderRadius: 16,
          padding: '22px 24px',
          color: 'var(--color-muted)',
          fontSize: 14,
        }}
      >
        <span style={{ width: 9, height: 9, borderRadius: 99, background: 'var(--color-teal)', animation: 'dundoBlink 1.1s infinite' }} />
        Listening for the artists you sound like — windowed embeddings over the Creative-Commons catalog…
      </div>
    </section>
  )
}

function ErrorNote({ msg }) {
  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '56px 28px 0' }}>
      <div
        style={{
          background: 'var(--color-paper)',
          border: '1px solid rgba(192,65,58,0.35)',
          borderRadius: 16,
          padding: '24px 26px',
        }}
      >
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#c0413a', marginBottom: 8 }}>
          Couldn&rsquo;t read this track
        </div>
        <p style={{ margin: 0, fontSize: 14.5, color: 'var(--color-ink-soft)' }}>{msg}</p>
      </div>
    </section>
  )
}
