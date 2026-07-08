/**
 * Hero — Dundo's signature: the sunset bloom (night → plum → magenta →
 * orange → gold) with a soft radial glow, echoing Suno's aurora look.
 * Tagline locked: "Upload an AI track. Find the indie artists it
 * resonates with." Full-bleed; text constrained to a centered column.
 */
export default function Hero() {
  return (
    <section
      style={{
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(135deg, #160B22 0%, #5B1E63 34%, #F5468A 64%, #FF7A3D 84%, #FFC24B 100%)',
        padding: '84px 28px 124px',
      }}
    >
      {/* soft luminescent bloom — the Suno-adjacent glow */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'radial-gradient(120% 90% at 78% 8%, rgba(255,194,75,0.45) 0%, rgba(255,122,61,0.18) 34%, transparent 62%), radial-gradient(90% 80% at 12% 100%, rgba(122,79,224,0.40) 0%, transparent 60%)',
          pointerEvents: 'none',
        }}
      />
      {/* bottom fade — dissolve the gradient into the page so there's no hard seam
          where the hero meets the dark body (and behind the floating drop zone) */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(to bottom, transparent 55%, var(--color-wash) 100%)',
          pointerEvents: 'none',
        }}
      />
      <div style={{ position: 'relative', maxWidth: 720, margin: '0 auto', textAlign: 'center', color: '#FBF3EC' }}>
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
