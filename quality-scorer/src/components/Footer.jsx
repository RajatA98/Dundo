/**
 * Footer — the trust-gap stance as first-class copy (PRD Goal 4 / Codex #8)
 * + the quiet suno/dundo Hindi pairing. Realizes the approved Dundo.dc.html
 * footer.
 */
export default function Footer() {
  return (
    <footer style={{ maxWidth: 940, margin: '0 auto', padding: '72px 28px 56px', width: '100%' }}>
      <div
        style={{
          borderTop: '1px solid var(--color-line)',
          paddingTop: 28,
          display: 'flex',
          gap: 28,
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
        }}
      >
        <p style={{ fontSize: 13, lineHeight: 1.6, color: 'var(--color-muted)', margin: 0, maxWidth: '56ch' }}>
          Dundo points you toward these artists — it doesn&rsquo;t train on them. Every match links straight to the
          artist&rsquo;s own page so you can support them.{' '}
          <a href="mailto:rajat1998@gmail.com?subject=Dundo%20removal%20request" style={{ color: 'var(--color-teal-deep)', textDecoration: 'none' }}>
            Artist? Request removal.
          </a>
        </p>
        <div style={{ fontSize: 13, color: 'var(--color-faint)', whiteSpace: 'nowrap' }}>
          suno (सुनो) listen · dundo (ढूँढो) find
        </div>
      </div>
    </footer>
  )
}
