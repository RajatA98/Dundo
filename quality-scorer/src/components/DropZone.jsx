import { useState } from 'react'

/**
 * DropZone — drag-and-drop or click-to-browse for an audio file.
 * Calls `onFile(File)` when a file is selected. Floats up over the hero band.
 *
 * @param {Object} props
 * @param {(file: File) => void} props.onFile
 * @param {boolean} [props.disabled=false]
 */
export default function DropZone({ onFile, disabled = false }) {
  const [dragging, setDragging] = useState(false)

  function pick(file) {
    if (!file || disabled) return
    onFile(file)
  }

  return (
    <div style={{ maxWidth: 940, margin: '-64px auto 0', padding: '0 28px' }}>
      <label
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          pick(e.dataTransfer.files?.[0])
        }}
        style={{
          display: 'block',
          background: 'var(--color-paper)',
          border: dragging ? '1.5px solid var(--color-teal)' : '1.5px solid var(--color-line)',
          borderRadius: 16,
          boxShadow: '0 18px 44px -22px rgba(14,17,22,0.22)',
          padding: '40px 32px',
          textAlign: 'center',
          cursor: disabled ? 'default' : 'pointer',
          opacity: disabled ? 0.6 : 1,
        }}
      >
        <input
          type="file"
          accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a"
          className="sr-only"
          disabled={disabled}
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <div
          style={{
            width: 52,
            height: 52,
            margin: '0 auto 18px',
            borderRadius: '50%',
            background: 'var(--color-teal-soft)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--color-teal-deep)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 16V4" />
            <path d="M7 9l5-5 5 5" />
            <path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
          </svg>
        </div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 500, marginBottom: 8 }}>
          {disabled ? 'Listening to your track…' : 'Drop your AI-generated track'}
        </div>
        <div style={{ fontSize: 13, color: 'var(--color-muted)', letterSpacing: '0.02em' }}>
          or click to browse · <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>mp3 wav flac ogg m4a</span>
        </div>
      </label>
    </div>
  )
}
