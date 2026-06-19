import { useEffect, useMemo, useRef, useState } from 'react'
import { audioUrlFor } from '../lib/api.js'

/**
 * SpectrogramCompare — the "Visual match" tab content.
 *
 * Two stacked WaveSurfer.js spectrograms (query on top, catalog match on
 * bottom) with the matched 10-second window highlighted on each. Per Codex
 * round-2 Q6: "show what the bars cannot" — time-local structure, energy
 * bands, vocal/instrument texture.
 *
 * WaveSurfer.js + Spectrogram plugin is lazy-imported on first mount to
 * keep it out of the initial bundle (~80 KB gzipped saved by deferring).
 *
 * Failure modes (handled gracefully):
 *   - No queryFile (replay after page refresh) → query pane shows fallback
 *   - No catalog audio URL (track lacks preview) → catalog pane shows fallback
 *   - WaveSurfer ESM module fails to load (network) → both panes show fallback
 */
export default function SpectrogramCompare({ queryFile, catalogTrack, matchTimestamp }) {
  const queryUrl = useMemo(() => {
    if (!queryFile) return null
    try {
      return URL.createObjectURL(queryFile)
    } catch {
      return null
    }
  }, [queryFile])

  // Revoke the blob URL on unmount to avoid leaking memory for large uploads.
  useEffect(() => {
    return () => {
      if (queryUrl) URL.revokeObjectURL(queryUrl)
    }
  }, [queryUrl])

  const catalogUrl = audioUrlFor(catalogTrack)

  const qs = Number(matchTimestamp?.queryStartSec)
  const qe = Number(matchTimestamp?.queryEndSec)
  const cs = Number(matchTimestamp?.catalogStartSec)
  const ce = Number(matchTimestamp?.catalogEndSec)
  const hasQueryWindow = Number.isFinite(qs) && Number.isFinite(qe) && qe > qs
  const hasCatalogWindow = Number.isFinite(cs) && Number.isFinite(ce) && ce > cs

  return (
    <div className="space-y-3 px-1 py-3">
      <SpectrogramPane
        label="Your upload"
        src={queryUrl}
        startSec={hasQueryWindow ? qs : null}
        endSec={hasQueryWindow ? qe : null}
        emptyNote="Upload not available for visualization — re-upload to render."
      />
      <SpectrogramPane
        label="Catalog match"
        src={catalogUrl}
        startSec={hasCatalogWindow ? cs : null}
        endSec={hasCatalogWindow ? ce : null}
        emptyNote="Preview audio unavailable for this catalog track."
      />
      <p
        className="font-mono text-[10px]"
        style={{ color: 'var(--color-faint)' }}
      >
        Highlighted band marks the matched 10-second window. Brighter regions
        in both spectrograms — especially in the same frequency range — are
        the visual evidence behind the timbre + harmonic agreement scores.
      </p>
    </div>
  )
}


function SpectrogramPane({ label, src, startSec, endSec, emptyNote }) {
  const containerRef = useRef(null)
  const wsRef = useRef(null)
  const [state, setState] = useState(src ? 'loading' : 'empty')

  useEffect(() => {
    if (!src || !containerRef.current) {
      setState(src ? 'loading' : 'empty')
      return
    }
    let cancelled = false
    let ws = null
    setState('loading')

    // Lazy-import WaveSurfer + the spectrogram plugin so the ~80 KB doesn't
    // hit the initial bundle. Dynamic import is cached after the first call,
    // so subsequent SpectrogramPanes mount instantly.
    Promise.all([
      import('wavesurfer.js'),
      import('wavesurfer.js/dist/plugins/spectrogram.esm.js'),
    ])
      .then(([{ default: WaveSurfer }, { default: SpectrogramPlugin }]) => {
        if (cancelled || !containerRef.current) return
        ws = WaveSurfer.create({
          container: containerRef.current,
          url: src,
          height: 32,
          waveColor: 'rgba(120, 120, 120, 0.4)',
          progressColor: 'rgba(160, 160, 160, 0.6)',
          interact: false,
          plugins: [
            SpectrogramPlugin.create({
              labels: false,
              height: 96,
              fftSamples: 512,
              splitChannels: false,
              scale: 'linear',
            }),
          ],
        })
        wsRef.current = ws
        ws.on('ready', () => {
          if (!cancelled) setState('ready')
        })
        ws.on('error', () => {
          if (!cancelled) setState('error')
        })
      })
      .catch(() => {
        if (!cancelled) setState('error')
      })

    return () => {
      cancelled = true
      try {
        if (ws) ws.destroy()
      } catch {
        /* swallow destroy errors during unmount */
      }
      wsRef.current = null
    }
  }, [src])

  // For the window overlay: compute the matched window as a percentage of the
  // full audio duration once WaveSurfer is ready. WaveSurfer doesn't tell us
  // duration until 'ready'; until then we render the band over the whole pane
  // (it'll snap to the correct width after load).
  const [duration, setDuration] = useState(null)
  useEffect(() => {
    if (state !== 'ready' || !wsRef.current) return
    try {
      const d = wsRef.current.getDuration()
      if (Number.isFinite(d) && d > 0) setDuration(d)
    } catch {
      /* ignore */
    }
  }, [state])

  const windowOverlay = useMemo(() => {
    if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || !duration) return null
    const leftPct = Math.max(0, Math.min(100, (startSec / duration) * 100))
    const widthPct = Math.max(0, Math.min(100 - leftPct, ((endSec - startSec) / duration) * 100))
    return { leftPct, widthPct }
  }, [startSec, endSec, duration])

  return (
    <div
      className="rounded-sm"
      style={{
        background: 'var(--color-bg)',
        border: '1px solid var(--color-line)',
      }}
    >
      <div className="flex items-center justify-between px-3 pt-2">
        <span
          className="font-mono text-[10px] uppercase"
          style={{
            color: 'var(--color-faint)',
            letterSpacing: '0.14em',
            fontWeight: 500,
          }}
        >
          {label}
        </span>
        <span
          className="font-mono text-[10px] tabular-nums"
          style={{ color: 'var(--color-faint)' }}
        >
          {Number.isFinite(startSec) && Number.isFinite(endSec)
            ? `match ${fmtSec(startSec)}–${fmtSec(endSec)}`
            : ''}
        </span>
      </div>

      {state === 'empty' && (
        <div
          className="px-3 py-6 font-mono text-[11px]"
          style={{ color: 'var(--color-faint)' }}
        >
          {emptyNote}
        </div>
      )}

      {state === 'error' && (
        <div
          className="px-3 py-6 font-mono text-[11px]"
          style={{ color: 'var(--color-faint)' }}
        >
          Spectrogram failed to render. The side-by-side snippet players above
          remain available.
        </div>
      )}

      {state !== 'empty' && state !== 'error' && (
        <div className="relative mx-3 my-2">
          <div ref={containerRef} className="w-full" />
          {windowOverlay && (
            <div
              className="pointer-events-none absolute top-0 bottom-0 rounded-sm"
              style={{
                left: `${windowOverlay.leftPct}%`,
                width: `${windowOverlay.widthPct}%`,
                background: 'rgba(255, 80, 80, 0.18)',
                border: '1px solid rgba(255, 80, 80, 0.45)',
              }}
              aria-hidden="true"
            />
          )}
          {state === 'loading' && (
            <div
              className="absolute inset-0 flex items-center justify-center font-mono text-[10px]"
              style={{ color: 'var(--color-faint)' }}
            >
              loading spectrogram…
            </div>
          )}
        </div>
      )}
    </div>
  )
}


function fmtSec(sec) {
  const n = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(n / 60)
  const r = n % 60
  return `${m}:${r.toString().padStart(2, '0')}`
}
