import { useState } from 'react'

const STATUS = {
  pass: { dot: 'bg-pass', text: 'text-pass', bar: 'bg-pass', label: 'PASS' },
  warn: { dot: 'bg-warn', text: 'text-warn', bar: 'bg-warn', label: 'WARN' },
  fail: { dot: 'bg-fail', text: 'text-fail', bar: 'bg-fail', label: 'FAIL' },
}

export default function SignalRow({ signal }) {
  const [open, setOpen] = useState(false)
  const s = STATUS[signal.status]
  const fill = Math.max(3, Math.round(signal.severity * 100))
  return (
    <div className="border-t border-line-soft py-3 first:border-t-0">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 text-left"
      >
        <span className={`h-2 w-2 shrink-0 ${s.dot}`} />
        <span className="w-12 shrink-0 font-mono text-[11px] text-faint">{signal.short}</span>
        <span className="flex-1 truncate text-sm text-ink">
          {signal.label}
          {signal.critical && (
            <span className="ml-2 align-middle font-mono text-[9px] uppercase tracking-wider text-faint">
              crit
            </span>
          )}
        </span>
        <span className={`shrink-0 font-mono text-sm tabular-nums ${s.text}`}>{signal.display}</span>
        <span className={`hidden w-10 shrink-0 text-right font-mono text-[10px] sm:inline ${s.text}`}>
          {s.label}
        </span>
        <svg
          className={`h-3 w-3 shrink-0 text-faint transition-transform ${open ? 'rotate-90' : ''}`}
          viewBox="0 0 12 12"
          aria-hidden="true"
        >
          <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" fill="none" />
        </svg>
      </button>

      <div className="ml-[1.75rem] mt-2 h-1 w-[calc(100%-1.75rem)] bg-line">
        <div className={`h-full ${s.bar}`} style={{ width: `${fill}%` }} />
      </div>

      {open && (
        <div className="ml-[1.75rem] mt-2 text-xs leading-relaxed text-dim">
          <p>{signal.blurb}</p>
          <p className="mt-1 font-mono text-[11px] text-faint">threshold · {signal.threshold}</p>
        </div>
      )}
    </div>
  )
}
