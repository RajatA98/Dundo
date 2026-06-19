import { Link } from 'react-router-dom'

/**
 * About page — brief in-app overview. The README on GitHub does the heavy
 * lifting (architecture, design decisions, run instructions, JOURNEY.md
 * for the chronological story). This page is the elevator pitch + pointers.
 */
export default function AboutPage() {
  return (
    <div className="mx-auto max-w-[68ch] py-16">
      <span
        className="block font-mono text-[12px] uppercase"
        style={{ color: 'var(--color-faint)', letterSpacing: '0.14em' }}
      >
        About
      </span>
      <h1
        className="mt-2"
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: '40px',
          fontWeight: 600,
          lineHeight: 1.1,
          letterSpacing: '-0.02em',
          color: 'var(--color-ink)',
        }}
      >
        Discover the indie artists your AI music resonates with.
      </h1>

      <div
        className="mt-8 space-y-5 text-[15px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        <p>
          Dundo (Hindi for <em>search</em>, paired with{' '}
          <strong style={{ color: 'var(--color-ink)' }}>Suno</strong>, Hindi for{' '}
          <em>listen</em>) takes an AI-generated track and retrieves indie
          human artists whose sound resonates with what you made. Each match
          comes with a grounded explanation of <em>why</em>, the underlying
          acoustic criteria (tempo, key, harmonic content, timbre), a
          side-by-side spectrogram view, and links to support the artist
          directly.
        </p>

        <p>
          The catalog is{' '}
          <strong style={{ color: 'var(--color-ink)' }}>Creative-Commons-licensed indie music</strong>{' '}
          (MTG-Jamendo today; Free Music Archive next). No major labels, no
          commercial-catalog ingestion, no copyright-detection framing — Dundo
          is positive-sum discovery, not policing.
        </p>

        <p>
          Each upload is embedded with an open-source audio model (MuQ-MuLan,
          512-d music-text joint embedding at 24 kHz), then nearest-neighbour
          searched against the local catalog. The discovery narrative is
          GPT-4o-mini consuming structured metadata — it does not hear audio,
          it does not determine copyright, and every citation it makes is
          validated against the supplied context before rendering.
        </p>

        <p>
          No black boxes. Retrieval quality is{' '}
          <Link
            to="/evaluation"
            style={{ color: 'var(--color-accent)', textDecoration: 'none' }}
          >
            measured, not claimed
          </Link>
          : Recall@1, Recall@3, MRR on a hand-built golden set, plus a top-1
          cosine histogram on unrelated negatives and named false-positive /
          false-negative examples with audio playback.
        </p>

        <p>
          Dundo is forked from{' '}
          <a
            href="https://github.com/RajatA98/PiedPiper"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'var(--color-accent)', textDecoration: 'none' }}
          >
            PiedPiper
          </a>
          , the original acoustic-similarity research project. See{' '}
          <a
            href="https://github.com/RajatA98/Dundo"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'var(--color-accent)', textDecoration: 'none' }}
          >
            project README on GitHub
          </a>{' '}
          for full architecture + decisions, or{' '}
          <code style={{ color: 'var(--color-ink)' }}>JOURNEY.md</code> for the
          chronological story of how we got here.
        </p>
      </div>
    </div>
  )
}
