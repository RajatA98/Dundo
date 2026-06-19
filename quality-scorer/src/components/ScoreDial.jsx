import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

function useCountUp(target, dur = 900) {
  const [n, setN] = useState(0)
  useEffect(() => {
    let raf
    let start
    const tick = (ts) => {
      if (start == null) start = ts
      const p = Math.min(1, (ts - start) / dur)
      const eased = 1 - Math.pow(1 - p, 3)
      setN(Math.round(target * eased))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, dur])
  return n
}

function bandColor(score) {
  if (score >= 85) return 'var(--color-pass)'
  if (score >= 60) return 'var(--color-warn)'
  return 'var(--color-fail)'
}

export default function ScoreDial({ score, size = 128 }) {
  const n = useCountUp(score)
  const r = size / 2 - 9
  const circ = 2 * Math.PI * r
  const arc = 0.75 // 270° sweep
  const dash = circ * arc
  const offset = dash * (1 - score / 100)
  const color = bandColor(score)
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-[135deg]">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--color-line)"
          strokeWidth="6"
          strokeDasharray={`${dash} ${circ}`}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={`${dash} ${circ}`}
          initial={{ strokeDashoffset: dash }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color}55)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-[2rem] font-semibold leading-none tabular-nums" style={{ color }}>
          {n}
        </span>
        <span className="kicker mt-1">/ 100</span>
      </div>
    </div>
  )
}
