import { NavLink, useLocation } from 'react-router-dom'

/**
 * Top nav — feather glyph + lowercase wordmark + three nav links.
 *
 * Visual contract: ui_mockup_v2_suno_flare.html `<nav class="topbar">`.
 * Wordmark is "pied piper" (lowercase, Outfit) per the Pied Piper visual
 * identity rules in CLAUDE_UI_DESIGN_PROMPT.md.
 */
export default function Nav() {
  const { pathname } = useLocation()
  const isHome = pathname === '/' || pathname === ''

  return (
    <nav
      className="sticky top-0 z-20 border-b"
      style={{ background: 'var(--color-bg)', borderColor: 'var(--color-line)' }}
    >
      <div className="mx-auto flex h-[72px] max-w-[1120px] items-center justify-between px-10">
        <NavLink to="/" className="flex select-none items-center gap-2.5">
          {/* Two-tone schematic feather glyph — Pied Piper 4.0 mark. */}
          <svg width="22" height="24" viewBox="0 0 22 24" fill="none" aria-hidden="true">
            <line x1="6" y1="22" x2="16.5" y2="3" stroke="var(--color-ink)" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="11" y1="13" x2="6" y2="11.5" stroke="var(--color-accent)" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="12.5" y1="10.2" x2="7.6" y2="8.5" stroke="var(--color-accent)" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="14" y1="7.3" x2="9.3" y2="5.6" stroke="var(--color-accent)" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="9.5" y1="15.8" x2="14.6" y2="14.3" stroke="var(--color-ink)" strokeWidth="1.6" strokeLinecap="round" />
            <line x1="11" y1="13" x2="16" y2="11.4" stroke="var(--color-ink)" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          <span
            style={{
              fontFamily: 'var(--font-wordmark)',
              fontWeight: 600,
              fontSize: '22px',
              letterSpacing: '-0.01em',
              color: 'var(--color-ink)',
              lineHeight: 1,
            }}
          >
            pied piper
          </span>
        </NavLink>

        <div className="flex items-center gap-6">
          <NavLinkItem to="/" exact active={isHome}>Home</NavLinkItem>
          <NavLinkItem to="/evaluation">Evaluation</NavLinkItem>
          <NavLinkItem to="/about">About</NavLinkItem>
        </div>
      </div>
    </nav>
  )
}

function NavLinkItem({ to, children, active, exact }) {
  return (
    <NavLink
      to={to}
      end={exact}
      className="cursor-pointer py-1 text-sm no-underline transition-colors"
      style={({ isActive }) => ({
        color: (isActive || active) ? 'var(--color-ink)' : 'var(--color-dim)',
        borderBottom: (isActive || active) ? '1px solid var(--color-accent)' : '1px solid transparent',
      })}
    >
      {children}
    </NavLink>
  )
}
