/**
 * Hero — kicker + tagline + subhead, center-aligned.
 *
 * The h1 tagline is locked verbatim (Richard Hendricks' original pilot
 * pitch, lightly modernized — see CLAUDE_UI_DESIGN_PROMPT.md). Don't
 * paraphrase.
 */
export default function Hero() {
  return (
    <header
      className="border-b py-16 text-center"
      style={{ borderColor: 'var(--color-line)' }}
    >
      <span
        className="block font-mono text-[11px] uppercase"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.18em',
          marginBottom: '24px',
        }}
      >
        Acoustic-similarity scanner for AI-generated music
      </span>
      <h1
        className="mx-auto text-balance"
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: '48px',
          lineHeight: 1.08,
          letterSpacing: '-0.025em',
          fontWeight: 600,
          maxWidth: '16ch',
          margin: '0 auto 16px',
          color: 'var(--color-ink)',
        }}
      >
        Find out if your AI-generated track resembles anything that&rsquo;s come before.
      </h1>
      <p
        className="mx-auto text-pretty leading-relaxed"
        style={{
          fontSize: '18px',
          color: 'var(--color-dim)',
          maxWidth: '60ch',
        }}
      >
        Drop in a Suno or Udio output. We embed the audio, compare it against a hand-curated catalog of real songs, and tell you which three you&rsquo;re closest to.
      </p>
    </header>
  )
}
