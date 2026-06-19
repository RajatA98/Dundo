export default function VerdictBadge({ verdict, size = 'md' }) {
  const keep = verdict === 'keep'
  const pad = size === 'lg' ? 'px-4 py-2 text-sm' : 'px-3 py-1.5 text-xs'
  return (
    <span
      className={`inline-flex items-center gap-2 border font-mono font-semibold uppercase tracking-[0.2em] ${pad} ${
        keep ? 'border-pass/50 bg-pass/10 text-pass' : 'border-fail/50 bg-fail/10 text-fail'
      }`}
    >
      <span className={`h-1.5 w-1.5 ${keep ? 'bg-pass' : 'bg-fail'}`} />
      {keep ? 'Keep' : 'Drop'}
    </span>
  )
}
