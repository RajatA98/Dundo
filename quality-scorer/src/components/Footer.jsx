import { Link } from 'react-router-dom'

/**
 * Footer — Easter-egg italic line + detector sigil row + repo / eval links.
 *
 * Visual contract: ui_mockup_v2_suno_flare.html `<footer>` + the `.detector-row`
 * block where only the Suno glyph is tinted rose.
 */
export default function Footer() {
  return (
    <footer
      className="mt-12 border-t py-10"
      style={{ borderColor: 'var(--color-line)' }}
    >
      <div className="mx-auto flex max-w-[1120px] flex-wrap items-center justify-between gap-6 px-10">
        <span
          className="text-[13px] italic"
          style={{ color: 'var(--color-faint)' }}
        >
          Originally pitched to a confused VC in 2014. Probably more useful now.
        </span>

        <DetectorRow />

        <div className="flex gap-6">
          <a
            href="https://github.com/RajatA98"
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-[12px] no-underline transition-colors"
            style={{ color: 'var(--color-dim)' }}
          >
            github ↗
          </a>
          <Link
            to="/evaluation"
            className="font-mono text-[12px] no-underline transition-colors"
            style={{ color: 'var(--color-dim)' }}
          >
            evaluation
          </Link>
        </div>
      </div>
    </footer>
  )
}

/**
 * Tiny "detects · S U ·" row. Only the Suno glyph is rose-tinted — a quiet
 * brag bar that doesn't take real estate. Used per the Suno-flare rule.
 */
function DetectorRow() {
  return (
    <span
      className="inline-flex items-center gap-2.5 font-mono text-[10px] uppercase"
      style={{
        color: 'var(--color-faint)',
        letterSpacing: '0.08em',
      }}
    >
      detects
      <SigilGlyph letter="S" suno />
      <SigilGlyph letter="U" title="Udio" />
      <SigilGlyph letter="·" title="Sonauto + others" />
    </span>
  )
}

function SigilGlyph({ letter, suno = false, title }) {
  return (
    <span
      title={title || (suno ? 'Suno' : '')}
      className="inline-flex items-center justify-center"
      style={{
        width: '16px',
        height: '16px',
        fontSize: '10px',
        fontWeight: 700,
        fontFamily: 'var(--font-wordmark)',
        borderRadius: '2px',
        border: suno
          ? '1px solid rgba(242, 92, 84, 0.35)'
          : '1px solid var(--color-line)',
        color: suno ? 'var(--color-suno-deep)' : 'var(--color-dim)',
        background: suno ? 'var(--color-suno-soft)' : 'transparent',
      }}
    >
      {letter}
    </span>
  )
}
