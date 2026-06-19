import { useEffect, useRef, useState } from 'react'
import { fmtDuration } from '../lib/format.js'

/**
 * Real audio transport with a single <audio> element per instance.
 *
 * When `src` is null, renders the disabled state — the user sees the control
 * but it's grayed out and clicks do nothing. This is intentional for catalog
 * rows that don't have a playable URL yet (e.g., Jamendo tracks before the
 * Phase 7.5 enrichment script runs).
 *
 * Cross-instance coordination: when any AudioPlayer starts playing, all other
 * <audio> elements on the page are paused. Implemented via a browser-native
 * 'play' event handler — no React context needed.
 */
export default function AudioPlayer({
  src = null,
  durationSec = 180,
  compact = false,
  artwork = null,
  size = 36,
  // ADR-0004 windowed-playback mode: when both are set, the component
  // plays only the [startSec, endSec) slice of the underlying audio.
  startSec = null,
  endSec = null,
}) {
  const audioRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [pos, setPos] = useState(0)
  const [realDuration, setRealDuration] = useState(durationSec)

  const disabled = !src
  const windowed = startSec != null && endSec != null && endSec > startSec

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const onTime = () => {
      setPos(audio.currentTime)
      // ADR-0004: hard-stop at endSec so the snippet truly plays only the window.
      if (windowed && audio.currentTime >= endSec) {
        audio.pause()
        audio.currentTime = startSec
      }
    }
    const onLoad = () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        setRealDuration(audio.duration)
      }
      // Seek to startSec on first load so a single click → play from the right place.
      if (windowed && Number.isFinite(audio.currentTime) && audio.currentTime < startSec) {
        audio.currentTime = startSec
      }
    }
    const onPlay = () => {
      setPlaying(true)
      // Pause every OTHER <audio> on the page so only one plays at a time.
      document.querySelectorAll('audio').forEach((a) => {
        if (a !== audio && !a.paused) a.pause()
      })
    }
    const onPause = () => setPlaying(false)
    const onEnded = () => {
      setPlaying(false)
      setPos(windowed ? startSec : 0)
    }

    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('loadedmetadata', onLoad)
    audio.addEventListener('play', onPlay)
    audio.addEventListener('pause', onPause)
    audio.addEventListener('ended', onEnded)

    return () => {
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('loadedmetadata', onLoad)
      audio.removeEventListener('play', onPlay)
      audio.removeEventListener('pause', onPause)
      audio.removeEventListener('ended', onEnded)
    }
  }, [src, windowed, startSec, endSec])

  const toggle = () => {
    const audio = audioRef.current
    if (!audio || disabled) return
    if (audio.paused) {
      // ADR-0004: when windowed, always seek to startSec before play so
      // re-clicking after a stop replays the snippet from the right place.
      if (windowed) {
        if (audio.currentTime < startSec || audio.currentTime >= endSec) {
          audio.currentTime = startSec
        }
      }
      const p = audio.play()
      if (p && typeof p.catch === 'function') p.catch(() => setPlaying(false))
    } else {
      audio.pause()
    }
  }

  const seek = (e) => {
    const audio = audioRef.current
    if (!audio || disabled || !realDuration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    audio.currentTime = ratio * realDuration
    setPos(audio.currentTime)
  }

  const pct = realDuration ? Math.min(100, (pos / realDuration) * 100) : 0

  // Compact + artwork → render artwork as the button itself with a play overlay.
  if (compact && artwork !== null) {
    return (
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-label={playing ? 'Pause' : (disabled ? 'No preview available' : 'Play preview')}
        className={`group relative shrink-0 overflow-hidden transition-opacity ${
          disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
        }`}
        style={{
          width: size,
          height: size,
          background: 'var(--color-elev)',
          border: '1px solid var(--color-line)',
        }}
        title={disabled ? 'No preview available' : (playing ? 'Pause' : 'Play preview')}
      >
        {src ? (
          <audio ref={audioRef} src={src} preload="metadata" crossOrigin="anonymous" />
        ) : null}
        {artwork ? (
          <img
            src={artwork}
            alt=""
            className="h-full w-full object-cover"
            loading="lazy"
            onError={(e) => { e.currentTarget.style.display = 'none' }}
          />
        ) : (
          // Placeholder: a musical-note glyph centered on the empty tile.
          <span
            className="absolute inset-0 grid place-items-center text-[14px]"
            style={{ color: 'var(--color-faint)' }}
          >
            ♪
          </span>
        )}
        {!disabled && (
          <span
            className={`absolute inset-0 grid place-items-center transition-opacity ${
              playing ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
            }`}
            style={{ background: 'rgba(14, 22, 32, 0.55)' }}
          >
            {playing ? (
              <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true" style={{ color: '#FAFAF7' }}>
                <rect x="2" y="1.5" width="3" height="9" fill="currentColor" />
                <rect x="7" y="1.5" width="3" height="9" fill="currentColor" />
              </svg>
            ) : (
              <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true" style={{ color: '#FAFAF7' }}>
                <path d="M2.5 1.5l8 4.5-8 4.5z" fill="currentColor" />
              </svg>
            )}
          </span>
        )}
      </button>
    )
  }

  // Compact, no artwork → simple icon-only button (existing behavior).
  const sizeBtn = compact ? 'h-7 w-7' : 'h-9 w-9'
  const sizeIcon = compact ? 10 : 12

  return (
    <div
      className="flex items-center gap-3"
      title={disabled ? 'No preview available' : (playing ? 'Pause' : 'Play preview')}
    >
      {src ? (
        <audio ref={audioRef} src={src} preload="metadata" crossOrigin="anonymous" />
      ) : null}
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        aria-label={playing ? 'Pause' : 'Play'}
        className={`grid ${sizeBtn} shrink-0 place-items-center border transition-colors ${
          disabled
            ? 'cursor-not-allowed opacity-40'
            : 'hover:border-accent hover:text-accent'
        }`}
        style={{
          borderColor: 'var(--color-line)',
          background: 'var(--color-elev)',
          color: 'var(--color-ink)',
        }}
      >
        {playing ? (
          <svg width={sizeIcon} height={sizeIcon} viewBox="0 0 12 12" aria-hidden="true">
            <rect x="2" y="1.5" width="3" height="9" fill="currentColor" />
            <rect x="7" y="1.5" width="3" height="9" fill="currentColor" />
          </svg>
        ) : (
          <svg width={sizeIcon} height={sizeIcon} viewBox="0 0 12 12" aria-hidden="true">
            <path d="M2.5 1.5l8 4.5-8 4.5z" fill="currentColor" />
          </svg>
        )}
      </button>
      {!compact && (
        <>
          <div
            className="relative h-1 flex-1 cursor-pointer"
            style={{ background: 'var(--color-line)' }}
            onClick={seek}
          >
            <div className="h-full" style={{ width: `${pct}%`, background: 'var(--color-accent)' }} />
          </div>
          <span
            className="shrink-0 font-mono text-[11px] tabular-nums"
            style={{ color: 'var(--color-faint)' }}
          >
            {fmtDuration(pos)} / {fmtDuration(realDuration)}
          </span>
        </>
      )}
    </div>
  )
}
