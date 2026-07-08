import { useState, useRef, useEffect } from 'react'
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
  const audioRef = useRef(null)
  const cardRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [docked, setDocked] = useState(false)
  const [progress, setProgress] = useState(0)

  // Dock the compact bar once the full card has scrolled above the viewport top,
  // so the player + stats stay reachable while browsing the matches below.
  useEffect(() => {
    const el = cardRef.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => setDocked(!entry.isIntersecting && entry.boundingClientRect.top < 0),
      { threshold: 0 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  const toggle = () => {
    const a = audioRef.current
    if (!a) return
    if (a.paused) a.play()
    else a.pause()
  }

  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '56px 28px 0' }}>
      <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 14 }}>
        Your track
      </div>
      <div ref={cardRef} style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 16, padding: '16px 18px' }}>
        <audio
          ref={audioRef}
          src={url}
          controls
          style={{ width: '100%', height: 38 }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={(e) => {
            const a = e.currentTarget
            setProgress(a.duration ? a.currentTime / a.duration : 0)
          }}
        />
        {stats && <SongStats stats={stats} />}
      </div>
      <StickyTrackBar docked={docked} playing={playing} progress={progress} stats={stats} onToggle={toggle} />
    </section>
  )
}

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const sync = () => setReduce(mq.matches)
    sync()
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])
  return reduce
}

/** Compact player that docks to the top of the viewport once the full "Your track"
 *  card scrolls away — play/pause + a condensed tempo · key · energy read-out, so
 *  the uploaded song stays one click away while scrolling the matches. */
function StickyTrackBar({ docked, playing, progress, stats, onToggle }) {
  const reduceMotion = usePrefersReducedMotion()
  const energyFill = ENERGY_FILL[stats?.energyBand] || 0
  const bits = []
  if (stats?.tempoBpm) bits.push(`≈ ${stats.tempoBpm} BPM`)
  if (stats?.key) bits.push(`${stats.key} ${stats.mode || ''}`.trim())

  return (
    <div
      aria-hidden={!docked}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        background: 'var(--color-paper)',
        boxShadow: docked ? '0 12px 34px -20px rgba(0,0,0,0.75)' : 'none',
        transform: docked ? 'translateY(0)' : 'translateY(-100%)',
        transition: reduceMotion ? 'none' : 'transform 0.26s ease',
        pointerEvents: docked ? 'auto' : 'none',
      }}
    >
      <div style={{ maxWidth: 940, margin: '0 auto', padding: '10px 28px', display: 'flex', alignItems: 'center', gap: 14 }}>
        <button
          onClick={onToggle}
          aria-label={playing ? 'Pause your track' : 'Play your track'}
          tabIndex={docked ? 0 : -1}
          style={{ flex: 'none', width: 34, height: 34, borderRadius: '50%', border: 'none', cursor: 'pointer', background: 'var(--color-teal)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          {playing ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="#fff"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="#fff" style={{ marginLeft: 2 }}><path d="M7 5l12 7-12 7z" /></svg>
          )}
        </button>
        <span style={{ flex: 'none', fontSize: 12, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)' }}>Your track</span>
        {(bits.length > 0 || stats?.energyBand) && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, fontFamily: 'var(--font-mono)', fontSize: 12.5, color: 'var(--color-faint)', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{bits.join(' · ')}</span>
            {stats?.energyBand && (
              <span style={{ flex: 'none', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                {bits.length > 0 && <span>·</span>}
                <span style={{ display: 'inline-flex', gap: 3 }}>
                  {[0, 1, 2, 3, 4].map((i) => (
                    <span key={i} style={{ width: 6, height: 6, borderRadius: 99, background: i < energyFill ? 'var(--color-teal)' : 'var(--color-line)' }} />
                  ))}
                </span>
              </span>
            )}
          </span>
        )}
      </div>
      {/* sunset underglow hairline — the signature echo */}
      <div aria-hidden="true" style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: 2, background: 'linear-gradient(90deg, #F5468A, #FF7A3D, #FFC24B)', opacity: 0.9 }} />
      {/* playback progress */}
      <div aria-hidden="true" style={{ position: 'absolute', left: 0, bottom: 0, height: 2, width: `${Math.round((progress || 0) * 100)}%`, background: 'var(--color-ink)', opacity: 0.35, transition: reduceMotion ? 'none' : 'width 0.1s linear' }} />
    </div>
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
          border: '1px solid rgba(255,107,107,0.38)',
          borderRadius: 16,
          padding: '24px 26px',
        }}
      >
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-fail)', marginBottom: 8 }}>
          Couldn&rsquo;t read this track
        </div>
        <p style={{ margin: 0, fontSize: 14.5, color: 'var(--color-ink-soft)' }}>{msg}</p>
      </div>
    </section>
  )
}
