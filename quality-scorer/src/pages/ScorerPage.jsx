import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Hero from '../components/Hero.jsx'
import DropZone from '../components/DropZone.jsx'
import ReportCard from '../components/ReportCard.jsx'
import { neighborsUpload, analyzeUpload } from '../lib/api.js'

/**
 * Landing page. Drop → top-3 similarity → ReportCard.
 *
 * Calls `/neighbors` (headline similarity) AND `/analyze` (quality badge) in
 * parallel. Quality badge is decoupled — it appears as soon as /analyze
 * returns, doesn't block the similarity headline if /analyze fails.
 *
 * Sequence of states: idle → analyzing → result / error.
 *
 * HF Space cold-start mitigation: the "warming up" copy swaps in at 6 s
 * elapsed (MuQ-MuLan loads on first request after sleep — ~45 s).
 */
export default function ScorerPage() {
  const [status, setStatus] = useState('idle')
  const [neighbors, setNeighbors] = useState(null)
  // ADR-0004: keep the upload File around so the SectionComparePanel can
  // play windowed slices of the user's own audio via createObjectURL.
  const [queryFile, setQueryFile] = useState(null)
  const [analyze, setAnalyze] = useState(null)
  const [error, setError] = useState('')

  const onFile = async (file) => {
    if (!file) return
    const ok = file.type.startsWith('audio/') || /\.(mp3|wav|flac|ogg|m4a)$/i.test(file.name)
    if (!ok) {
      setNeighbors(null)
      setAnalyze(null)
      setError(`Couldn't read "${file.name}" — expected an audio file.`)
      setStatus('error')
      return
    }
    setError('')
    setNeighbors(null)
    setAnalyze(null)
    setQueryFile(file)
    setStatus('analyzing')

    // Fire both calls in parallel. /neighbors is the headline; /analyze
    // populates the quality badge. /analyze failures don't block the report.
    const analyzeP = analyzeUpload(file).then(
      (r) => { setAnalyze(r); return r },
      () => null,
    )
    try {
      const n = await neighborsUpload(file, 3)
      setNeighbors(n)
      setStatus('result')
      analyzeP // continues in background; updates state when it resolves
    } catch (e) {
      setError(`Couldn't analyze "${file.name}" — ${e.message || 'unknown error'}.`)
      setStatus('error')
    }
  }

  return (
    <>
      <Hero />

      <section className="py-10">
        <DropZone onFile={onFile} disabled={status === 'analyzing'} />
      </section>

      <section className="py-2 pb-16">
        <div
          className="mb-6 font-mono text-[12px] uppercase"
          style={{ color: 'var(--color-faint)', letterSpacing: '0.14em' }}
        >
          Report
        </div>

        <AnimatePresence mode="wait">
          {status === 'analyzing' && <Analyzing key="a" />}
          {status === 'error' && <ErrorState key="e" msg={error} />}
          {status === 'result' && neighbors && (
            <ReportCard
              key={neighbors.query?.id || 'result'}
              neighbors={neighbors}
              analyze={analyze}
              queryFile={queryFile}
            />
          )}
          {status === 'idle' && <IdleState key="i" />}
        </AnimatePresence>
      </section>
    </>
  )
}

function Analyzing() {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const start = Date.now()
    const id = setInterval(() => setElapsed(Date.now() - start), 500)
    return () => clearInterval(id)
  }, [])
  const warming = elapsed > 6000
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="p-10"
      style={{
        background: 'var(--color-bg)',
        border: '1px solid var(--color-line)',
        borderRadius: '4px',
      }}
    >
      <div
        className="flex items-center gap-3 font-mono text-sm"
        style={{ color: 'var(--color-dim)' }}
      >
        <span
          className="block h-2 w-2"
          style={{ background: 'var(--color-accent)', borderRadius: '2px', animation: 'pp-blink 1.2s infinite' }}
        />
        {warming
          ? 'warming up the analyzer (first request after idle takes ~45 s)…'
          : 'analyzing · windowed MuQ-MuLan embeddings + cosine sweep over the reference catalog…'}
      </div>
      <style>{`@keyframes pp-blink { 0%, 100% { opacity: 1 } 50% { opacity: 0.3 } }`}</style>
    </motion.div>
  )
}

function ErrorState({ msg }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8"
      style={{
        border: '1px solid rgba(199, 57, 54, 0.40)',
        background: 'rgba(199, 57, 54, 0.05)',
        borderRadius: '4px',
      }}
    >
      <div className="flex items-start gap-3">
        <span
          className="mt-1 block h-2 w-2 shrink-0"
          style={{ background: 'var(--color-fail)' }}
        />
        <div>
          <div
            className="font-mono text-xs uppercase"
            style={{ color: 'var(--color-fail)', letterSpacing: '0.1em' }}
          >
            Couldn’t read this track
          </div>
          <p className="mt-2 text-sm" style={{ color: 'var(--color-ink)' }}>{msg}</p>
          <p className="mt-1 text-xs" style={{ color: 'var(--color-dim)' }}>
            Supported: mp3, wav, flac, ogg, m4a.
          </p>
        </div>
      </div>
    </motion.div>
  )
}

function IdleState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="p-10 text-center"
      style={{
        border: '1px dashed var(--color-line)',
        background: 'transparent',
        borderRadius: '4px',
      }}
    >
      <p
        className="font-mono text-sm"
        style={{ color: 'var(--color-faint)' }}
      >
        No track loaded — drop a file above to get started.
      </p>
    </motion.div>
  )
}
