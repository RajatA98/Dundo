import { motion } from 'framer-motion'

const BAR_COLOR = {
  clip: 'bg-fail',
  silence: 'bg-faint/45',
  truncation: 'bg-warn',
}

function barType(i, problems) {
  for (const p of problems) if (i >= p.from && i < p.to) return p.type
  return null
}

// Centered peak-envelope view; fault regions are lit by type, like a scope flagging faults.
export default function Waveform({ peaks, problems = [], height = 96, animate = true, className = '' }) {
  return (
    <div className={`relative w-full ${className}`} style={{ height }}>
      {/* center baseline */}
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-line" />
      <motion.div
        className="relative flex h-full w-full items-center gap-px"
        initial={animate ? { clipPath: 'inset(0 100% 0 0)' } : false}
        animate={{ clipPath: 'inset(0 0% 0 0)' }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
      >
        {peaks.map((v, i) => {
          const t = barType(i, problems)
          const color = t ? BAR_COLOR[t] : 'bg-accent/55'
          return (
            <div
              key={i}
              className={`flex-1 rounded-[1px] ${color}`}
              style={{ height: `${Math.max(2, v * 100)}%` }}
            />
          )
        })}
      </motion.div>
    </div>
  )
}
