import { Link, useLocation } from 'react-router-dom'

/** Dundo nav — `Dundo.` wordmark (teal dot) + About / Evaluation links. */
export default function Nav() {
  const { pathname } = useLocation()
  const evalActive = pathname.startsWith('/evaluation')
  return (
    <nav
      className="mx-auto flex w-full max-w-[1080px] items-center justify-between"
      style={{ padding: '22px 28px' }}
    >
      <Link
        to="/"
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: 25,
          fontWeight: 600,
          letterSpacing: '-0.01em',
          color: 'var(--color-ink)',
          textDecoration: 'none',
          lineHeight: 1,
        }}
      >
        Dundo<span style={{ color: 'var(--color-teal)' }}>.</span>
      </Link>
      <div className="flex items-center" style={{ gap: 28, fontSize: 14.5, fontWeight: 500 }}>
        <Link to="/about" style={{ color: 'var(--color-muted)', textDecoration: 'none' }}>
          About
        </Link>
        <Link
          to="/evaluation"
          style={{ color: evalActive ? 'var(--color-teal-deep)' : 'var(--color-muted)', textDecoration: 'none' }}
        >
          Evaluation
        </Link>
      </div>
    </nav>
  )
}
