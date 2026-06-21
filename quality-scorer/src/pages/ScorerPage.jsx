import { useState } from 'react'
import Hero from '../components/Hero.jsx'
import DropZone from '../components/DropZone.jsx'
import ArtistResults, { EmptyState } from '../components/ArtistResults.jsx'
import { sampleArtists } from '../lib/sampleArtists.js'

/**
 * Landing page — drop an AI track, meet the top-3 indie artists who sound
 * like it. Realizes the approved Dundo.dc.html app view: hero → drop zone →
 * artist results (Case A) → the honest Case-B state.
 *
 * ┌──────────────────────────────────────────────────────────────────────┐
 * │ ⚠ PROTOTYPE / PREVIEW — NOT wired to the backend.                      │
 * │ The drop zone does NOT call /neighbors or /analyze. It runs a fake     │
 * │ "listening" beat and renders the static `sampleArtists` regardless of  │
 * │ the file. This is the Phase-5 UI shell over the frozen artist contract;│
 * │ Phase 3 replaces the setTimeout below with a real `neighborsUpload()`  │
 * │ call returning the artist-framed response (same component contract).   │
 * └──────────────────────────────────────────────────────────────────────┘
 */
export default function ScorerPage() {
  const [phase, setPhase] = useState('results') // 'results' | 'analyzing' | 'error'
  const [results, setResults] = useState(sampleArtists)
  const [error, setError] = useState('')

  const onFile = (file) => {
    if (!file) return
    const ok = file.type.startsWith('audio/') || /\.(mp3|wav|flac|ogg|m4a)$/i.test(file.name)
    if (!ok) {
      setError(`Couldn't read "${file.name}" — expected an audio file (mp3, wav, flac, ogg, m4a).`)
      setPhase('error')
      return
    }
    setError('')
    setPhase('analyzing')
    // PROTOTYPE: no backend call — Phase 3 replaces this with
    // `neighborsUpload(file, 3)` → artist-framed response.
    if (import.meta.env.DEV) {
      console.info('[Dundo] preview mode: rendering sampleArtists, not real /neighbors results (Phase 3 wires this).')
    }
    setTimeout(() => {
      setResults(sampleArtists)
      setPhase('results')
    }, 1100)
  }

  return (
    <>
      <Hero />
      <DropZone onFile={onFile} disabled={phase === 'analyzing'} />

      {phase === 'analyzing' && <Analyzing />}
      {phase === 'error' && <ErrorNote msg={error} />}
      {phase === 'results' && (
        <>
          <ArtistResults artists={results} />
          {/* Case-B preview, mirroring the approved design canvas. */}
          <EmptyState asPreview />
        </>
      )}
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
