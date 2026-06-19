import { useRef, useState } from 'react'

/**
 * DropZone — drag-and-drop or click-to-browse for an audio file.
 *
 * Calls `onFile(File)` when a file is selected. Visual contract:
 * ui_mockup_v2_suno_flare.html `.dropzone` block.
 *
 * @param {Object} props
 * @param {(file: File) => void} props.onFile
 * @param {boolean} [props.disabled=false] - dims + ignores events when true
 */
export default function DropZone({ onFile, disabled = false }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  function pick(file) {
    if (!file || disabled) return
    onFile(file)
  }

  return (
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
      className="flex min-h-[280px] cursor-pointer flex-col items-center justify-center gap-2 p-10 transition-colors"
      style={{
        border: dragging
          ? '1px dashed var(--color-accent)'
          : '1px dashed #C9CABF',
        background: dragging ? 'var(--color-accent-dim)' : 'transparent',
        borderRadius: '4px',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a"
        className="sr-only"
        disabled={disabled}
        onChange={(e) => pick(e.target.files?.[0])}
      />
      <div className="mb-2">
        <svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">
          <rect x="6.5" y="6.5" width="21" height="21" rx="2" stroke="var(--color-faint)" strokeWidth="1.4" />
          <line x1="17" y1="12" x2="17" y2="22" stroke="var(--color-accent)" strokeWidth="1.6" strokeLinecap="round" />
          <line x1="12" y1="17" x2="22" y2="17" stroke="var(--color-accent)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </div>
      <div className="text-[18px] font-medium" style={{ color: 'var(--color-ink)' }}>
        Drop an audio file
      </div>
      <div
        className="font-mono text-[12px]"
        style={{
          color: 'var(--color-faint)',
          letterSpacing: '0.02em',
        }}
      >
        or click to browse · mp3 wav flac m4a
      </div>
    </label>
  )
}
