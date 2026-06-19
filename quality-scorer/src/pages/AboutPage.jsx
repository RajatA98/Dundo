import { Link } from 'react-router-dom'

/**
 * About page — brief in-app overview. The README on GitHub does the heavy
 * lifting (architecture, design decisions, "what I left out", run instructions).
 * This page is just the elevator pitch + a couple of pointers.
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
        An honest audio-embedding pipeline with a published eval.
      </h1>

      <div
        className="mt-8 space-y-5 text-[15px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        <p>
          PiedPiper embeds each track with an open-source audio model
          (MuQ-MuLan, 512-d music-text joint embedding at 24 kHz), then runs
          nearest-neighbour search against a hand-curated reference catalog of
          roughly{' '}
          <strong style={{ color: 'var(--color-ink)' }}>100 recognizable tracks</strong>{' '}
          from iTunes previews and{' '}
          <strong style={{ color: 'var(--color-ink)' }}>150+ Creative Commons</strong>{' '}
          songs from MTG-Jamendo. Two independent ACRCloud signals — cover-song ID and
          AI music detection — run alongside as commercial second opinions.
        </p>

        <p>
          No black boxes. The detector quality is{' '}
          <Link
            to="/evaluation"
            style={{ color: 'var(--color-accent)', textDecoration: 'none' }}
          >
            measured, not claimed
          </Link>
          : Recall@1, Recall@3, MRR on a hand-built golden set, plus a top-1 cosine
          histogram on unrelated negatives and named false-positive / false-negative
          examples with audio playback.
        </p>

        <p>
          The reference catalog is a sampled demo set, not a production catalog —
          productionizing this would mean indexing a licensed catalog the way a vendor
          would internally. That trade-off is named explicitly on the evaluation page
          under{' '}
          <em style={{ color: 'var(--color-ink)' }}>limitations</em>.
        </p>

        <p>
          See the full architecture and the &ldquo;what I deliberately left out&rdquo; section in
          the{' '}
          <a
            href="https://github.com/RajatA98"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'var(--color-accent)', textDecoration: 'none' }}
          >
            project README on GitHub
          </a>
          .
        </p>
      </div>
    </div>
  )
}
