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
          comes with a grounded explanation of <em>why</em>, the shared sonic
          descriptors behind it, a side-by-side spectrogram view, and links to
          give the artist a listen and support them directly.
        </p>

        <p>
          The catalog is{' '}
          <strong style={{ color: 'var(--color-ink)' }}>Creative-Commons-licensed indie music</strong>{' '}
          (MTG-Jamendo today; Free Music Archive next). No major labels, no
          commercial-catalog ingestion, no copyright-detection framing — Dundo
          is positive-sum discovery, not policing.
        </p>

        <p>
          The matches themselves are <strong style={{ color: 'var(--color-ink)' }}>deterministic
          and content-based</strong>: each upload is embedded with an open-source
          audio model (MuQ-MuLan, a 512-d music-text joint embedding), then
          nearest-neighbour searched against the catalog with FAISS — the same
          track always returns the same artists, with no LLM in the loop. Only
          the <em>explanation</em> is generated: a retrieval-augmented layer feeds
          GPT-4o-mini structured facts about the already-decided match — the
          shared descriptors and what the catalog knows about the artist. It
          never hears the audio and never decides a match, and every claim it
          makes — from a cited number to an artist's location — is validated
          against those facts before rendering. Anything unsupported is dropped,
          not shown.
        </p>

        <p>
          No black boxes. Retrieval quality is{' '}
          <Link
            to="/evaluation"
            style={{ color: 'var(--color-accent)', textDecoration: 'none' }}
          >
            measured, not claimed
          </Link>
          : Recall@1, Recall@3, and MRR by leave-one-out over the live catalog,
          plus a top-1 cosine histogram showing the noise floor on unrelated
          tracks. The narrative layer has its own gate — a 16-case eval that
          must reject every hallucinated citation before a build ships.
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
