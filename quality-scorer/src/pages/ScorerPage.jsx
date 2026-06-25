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
  const [error, setError] = useState('')

  const onFile = async (file) => {
    if (!file) return
    const ok = file.type.startsWith('audio/') || /\.(mp3|wav|flac|ogg|m4a)$/i.test(file.name)
    if (!ok) {
      setError(`Couldn't read "${file.name}" — expected an audio file (mp3, wav, flac, ogg, m4a).`)
      setPhase('error')
      return
    }
    setError('')
    setPhase('analyzing')
    try {
      const res = await neighborsUpload(file, 3)
      const m = Array.isArray(res?.matches) ? res.matches : []
      setMatches(m)
      setContextToken(res?.contextToken || null)
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
      {phase === 'results' && <ArtistResults artists={matches} contextToken={contextToken} />}
      {phase === 'empty' && <EmptyState />}
    </>
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
