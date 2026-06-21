/**
 * Hero — the violet-free, cool teal→indigo signature band (Dundo's one
 * gradient). Tagline locked: "Upload an AI track. Find the indie artists
 * it resonates with." Full-bleed; text constrained to a centered column.
 */
export default function Hero() {
  return (
    <section style={{ background: 'linear-gradient(150deg, #0c8f86, #3A57D6)', padding: '84px 28px 124px' }}>
      <div style={{ maxWidth: 720, margin: '0 auto', textAlign: 'center', color: '#ffffff' }}>
        <div
          style={{
            fontSize: 12.5,
            fontWeight: 600,
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: 'rgba(255,255,255,0.82)',
            marginBottom: 22,
          }}
        >
          Discovery for AI-music creators
        </div>
        <h1
          style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 500,
            fontSize: 47,
            lineHeight: 1.1,
            letterSpacing: '-0.015em',
            margin: '0 0 22px',
          }}
        >
          Upload an AI track. Find the indie artists it resonates with.
        </h1>
        <p style={{ fontSize: 17.5, lineHeight: 1.55, color: 'rgba(255,255,255,0.88)', margin: '0 auto', maxWidth: 560 }}>
          Drop a track you made with Suno or Udio — meet the real indie artists who sound like it, and give them a listen.
        </p>
      </div>
    </section>
  )
}
