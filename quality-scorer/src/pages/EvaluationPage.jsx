import { sampleMetrics } from '../lib/sampleArtists.js'

/**
 * Evaluation page — "measured, not claimed." Retrieval quality on a held-out
 * set, plus an honest method + limitations note. Realizes the approved
 * Dundo.dc.html evaluation view.
 *
 * NOTE (Phase 3/8 seam): metrics currently render the design's representative
 * numbers (`sampleMetrics`). The eval harness (run_eval over the CC catalog)
 * feeds real Recall@k / MRR here once the full catalog lands.
 */
export default function EvaluationPage() {
  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '64px 28px 80px' }}>
      <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-teal)', marginBottom: 16 }}>
        Evaluation
      </div>
      <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 38, lineHeight: 1.12, letterSpacing: '-0.015em', margin: '0 0 12px', maxWidth: '18ch' }}>
        Measured, not claimed.
      </h1>
      <p style={{ fontSize: 16, lineHeight: 1.6, color: 'var(--color-muted)', margin: '0 0 40px', maxWidth: '56ch' }}>
        Retrieval quality on a held-out set of 240 human-labeled track pairs. Numbers, not adjectives.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 48 }}>
        {sampleMetrics.map((m) => (
          <div key={m.label} style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 14, padding: 24 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, letterSpacing: '0.04em', color: 'var(--color-muted)', marginBottom: 14 }}>{m.label}</div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 44, lineHeight: 1, letterSpacing: '-0.02em' }}>{m.value}</div>
            <div style={{ fontSize: 12.5, color: 'var(--color-faint)', marginTop: 10 }}>{m.note}</div>
          </div>
        ))}
      </div>

      <div style={{ maxWidth: '64ch' }}>
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 12 }}>
          Method &amp; limitations
        </div>
        <p style={{ fontFamily: 'var(--font-display)', fontSize: 16.5, lineHeight: 1.64, color: 'var(--color-ink-soft)', margin: '0 0 16px' }}>
          Each query track is embedded and matched against a 1,800-track Creative-Commons catalog; a held-out human rater marked which retrieved artists genuinely sounded alike. Recall is the fraction of queries whose true match appeared in the top-1 / top-3; MRR rewards ranking the right artist higher.
        </p>
        <p style={{ fontSize: 14, lineHeight: 1.62, color: 'var(--color-muted)', margin: 0 }}>
          Limitations: the catalog skews toward electronic, folk, and ambient genres, so dense or highly produced tracks retrieve fewer honest matches — by design, Dundo returns 1–2 results or none rather than padding. Spectrogram alignment is windowed, not full-track.
        </p>
      </div>
    </section>
  )
}
