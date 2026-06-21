import ArtistCard from './ArtistCard.jsx'

/**
 * ArtistResults — Case A: the top-3 (or fewer honest) artist matches.
 * The first card defaults to expanded (as in the approved design) so the
 * evidence is visible without a click. Never padded — renders exactly what
 * it's given (FR-9).
 *
 * @param {{ artists: object[] }} props
 */
export default function ArtistResults({ artists }) {
  if (!artists || artists.length === 0) return null
  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '64px 28px 0' }}>
      <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-teal)', marginBottom: 28 }}>
        The artists you sound like
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
        {artists.map((artist, i) => (
          <ArtistCard key={artist.artistId} artist={artist} defaultExpanded={i === 0} />
        ))}
      </div>
    </section>
  )
}

/**
 * EmptyState — Case B: nothing crossed the threshold. Honest, calm, not red,
 * no placeholder rows. Shown instead of (or below) results.
 *
 * @param {{ asPreview?: boolean }} props - asPreview renders the "when nothing
 *   is a close match" divider above it (mirrors the approved design canvas).
 */
export function EmptyState({ asPreview = false }) {
  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: asPreview ? '48px 28px 0' : '64px 28px 0' }}>
      {asPreview && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 28 }}>
          <div style={{ flex: 1, height: 1, background: 'var(--color-line)' }} />
          <span style={{ fontSize: 11.5, letterSpacing: '0.06em', color: 'var(--color-faint)' }}>when nothing is a close match</span>
          <div style={{ flex: 1, height: 1, background: 'var(--color-line)' }} />
        </div>
      )}
      <div style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 16, padding: '56px 32px', textAlign: 'center' }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 28, lineHeight: 1.18, letterSpacing: '-0.01em', margin: '0 auto 14px', maxWidth: '28ch' }}>
          Nothing in our catalog is a close match — yet.
        </h2>
        <p style={{ fontSize: 15.5, lineHeight: 1.6, color: 'var(--color-muted)', margin: '0 auto', maxWidth: '42ch' }}>
          Your sound is distinct. Try another track, or check back as the catalog grows.
        </p>
      </div>
    </section>
  )
}
