import { useMemo, useState } from 'react'
import { simLabel } from '../lib/sampleArtists.js'

/**
 * ArtistCard — the hero of the results. The human artist leads; the cosine
 * similarity is a quiet secondary bar, never a loud %. Realizes the approved
 * Dundo.dc.html card. Binds to the frozen artist contract (ArtistMatch).
 *
 * Optional fields (location, supportLinks, spotifyUrl) render ONLY when present
 * — never an empty placeholder (FR-5). Orange is reserved for "give them a
 * listen"; teal carries the find identity.
 *
 * @param {{ artist: object, defaultExpanded?: boolean }} props
 */
export default function ArtistCard({ artist, defaultExpanded = false }) {
  const [playing, setPlaying] = useState(false)
  const [expanded, setExpanded] = useState(defaultExpanded)

  const hasLocation = !!artist.location
  const hasSupport = artist.supportLinks && artist.supportLinks.length > 0
  const hasSpotify = !!artist.spotifyUrl
  const expandable = artist.criteria && artist.criteria.length > 0
  const sim = Math.round((artist.similarity ?? 0) * 100)

  // Seeded bars so the waveform is stable per artist (mirrors the design).
  const bars = useMemo(() => {
    const out = []
    let s = (artist.artistId || '').split('').reduce((a, c) => a + c.charCodeAt(0), 7) * 137
    for (let i = 0; i < 44; i++) {
      s = (s * 9301 + 49297) % 233280
      out.push(18 + Math.round((s / 233280) * 78))
    }
    return out
  }, [artist.artistId])

  return (
    <div style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 16, padding: 28 }}>
      {/* top row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 18 }}>
        <div style={{ flex: 'none', width: 76, height: 76, borderRadius: 12, background: artist.artGrad || 'linear-gradient(140deg,#0c8f86,#3A57D6)', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', inset: 0, backgroundImage: 'repeating-linear-gradient(135deg, rgba(255,255,255,0.10) 0 7px, transparent 7px 14px)' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0, paddingTop: 2 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
            <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 27, lineHeight: 1.1, letterSpacing: '-0.01em', margin: 0 }}>
              {artist.name}
            </h3>
            {hasLocation && (
              <span style={{ fontSize: 13, color: 'var(--color-muted)', whiteSpace: 'nowrap' }}>based in {artist.location}</span>
            )}
          </div>
          {/* quiet similarity signal */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
            <div style={{ width: 96, height: 4, borderRadius: 99, background: 'var(--color-line)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${sim}%`, background: 'var(--color-teal)', borderRadius: 99 }} />
            </div>
            <span style={{ fontSize: 11.5, letterSpacing: '0.04em', color: 'var(--color-muted)' }}>{simLabel(artist.similarity ?? 0)}</span>
          </div>
        </div>
      </div>

      {/* audio preview */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 22, padding: '12px 16px', background: 'var(--color-wash)', borderRadius: 12 }}>
        <button
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? 'Pause preview' : 'Play preview'}
          style={{ flex: 'none', width: 38, height: 38, borderRadius: '50%', border: 'none', cursor: 'pointer', background: 'var(--color-ink)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          {playing ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="#fff"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="#fff" style={{ marginLeft: 2 }}><path d="M7 5l12 7-12 7z" /></svg>
          )}
        </button>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 3, height: 34 }}>
          {bars.map((h, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${h}%`,
                minHeight: 3,
                borderRadius: 2,
                background: playing ? 'var(--color-teal-deep)' : '#c3ccd1',
                transformOrigin: 'center',
                animation: playing ? 'dundoEq 0.9s ease-in-out infinite' : 'none',
                animationDelay: playing ? `${(i % 8) * 0.06}s` : undefined,
              }}
            />
          ))}
        </div>
        <span style={{ flex: 'none', fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--color-muted)' }}>{artist.duration || ''}</span>
      </div>

      {/* why this resonates */}
      <div style={{ marginTop: 22 }}>
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-teal)', marginBottom: 9 }}>
          Why this resonates
        </div>
        <p style={{ fontFamily: 'var(--font-display)', fontWeight: 400, fontSize: 17, lineHeight: 1.62, color: 'var(--color-ink-soft)', margin: 0, maxWidth: '64ch' }}>
          {artist.narrative}
        </p>
      </div>

      {/* action row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginTop: 24 }}>
        <a
          href={artist.listenUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ display: 'inline-flex', alignItems: 'center', gap: 7, background: 'var(--color-listen)', color: '#fff', fontSize: 14.5, fontWeight: 600, textDecoration: 'none', padding: '11px 18px', borderRadius: 11 }}
        >
          Give them a listen <span style={{ fontSize: 15 }}>↗</span>
        </a>

        {hasSupport && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {artist.supportLinks.map((link) => (
              <a
                key={link.url}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                style={{ display: 'inline-flex', alignItems: 'center', fontSize: 13, fontWeight: 500, color: 'var(--color-teal-deep)', textDecoration: 'none', padding: '9px 13px', border: '1px solid var(--color-line)', borderRadius: 999 }}
              >
                {link.label}
              </a>
            ))}
          </div>
        )}

        {hasSpotify && (
          <a
            href={artist.spotifyUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: '#1DB954', color: '#fff', fontSize: 13.5, fontWeight: 600, textDecoration: 'none', padding: '9px 15px 9px 12px', borderRadius: 999 }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff" aria-hidden="true"><path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.5 17.31a.75.75 0 0 1-1.03.25c-2.82-1.72-6.37-2.11-10.56-1.16a.75.75 0 1 1-.33-1.46c4.58-1.05 8.51-.6 11.67 1.33.35.22.47.69.25 1.04zm1.47-3.27a.94.94 0 0 1-1.29.31c-3.23-1.99-8.15-2.56-11.97-1.4a.94.94 0 1 1-.54-1.8c4.37-1.32 9.79-.68 13.5 1.6.44.27.58.85.3 1.29zm.13-3.4C15.7 8.3 9.1 8.07 5.4 9.2a1.12 1.12 0 1 1-.65-2.15c4.25-1.29 11.54-1.04 16.1 1.66a1.12 1.12 0 1 1-1.15 1.93z" /></svg>
            Listen on Spotify
          </a>
        )}

        {expandable && (
          <button
            onClick={() => setExpanded((e) => !e)}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', font: 'inherit', fontSize: 13, fontWeight: 500, color: 'var(--color-muted)', display: 'inline-flex', alignItems: 'center', gap: 5 }}
          >
            {expanded ? 'Hide the evidence' : 'See the evidence'}
            <span style={{ display: 'inline-block', transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: 11 }}>▾</span>
          </button>
        )}
      </div>

      {/* expanded evidence */}
      {expandable && expanded && (
        <div style={{ marginTop: 26, paddingTop: 24, borderTop: '1px solid var(--color-line)' }}>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 18 }}>
            The evidence behind the match
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px 36px' }}>
            {artist.criteria.map((crit) => (
              <div key={crit.label}>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, marginBottom: 7 }}>
                  <span style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-muted)' }}>{crit.label}</span>
                  <span style={{ fontSize: 13, color: 'var(--color-ink-soft)' }}>{crit.detail}</span>
                </div>
                <div style={{ height: 4, borderRadius: 99, background: 'var(--color-line)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.round((crit.agreement ?? 0) * 100)}%`,
                      background: (crit.agreement ?? 0) >= 0.6 ? 'var(--color-teal)' : 'var(--color-indigo)',
                      borderRadius: 99,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          {artist.spectro && artist.spectro.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <div style={{ display: 'flex', gap: 16 }}>
                {artist.spectro.map((sp) => (
                  <div key={sp.caption} style={{ flex: 1 }}>
                    <div style={{ height: 92, borderRadius: 10, backgroundColor: '#0E1116', backgroundImage: 'repeating-linear-gradient(90deg, rgba(15,181,166,0.55) 0 2px, transparent 2px 5px), repeating-linear-gradient(0deg, rgba(58,87,214,0.30) 0 2px, transparent 2px 9px)' }} />
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-muted)', marginTop: 8 }}>{sp.caption}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 12.5, color: 'var(--color-muted)', marginTop: 12, lineHeight: 1.5 }}>
                Matched ~10-second windows, aligned. The shared energy in the low-mids and the matching harmonic spacing are what surfaced this pairing.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
