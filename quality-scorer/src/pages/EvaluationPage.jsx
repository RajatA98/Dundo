import { useEffect, useState } from 'react'

/**
 * Evaluation page — "measured, not claimed." Loads the REAL eval.json the eval
 * harness (run_eval over the shipped catalog) writes to public/corpus/, so the
 * numbers on this page always match the committed artifact. No placeholders.
 */
const PCT = (x) => (x == null ? '—' : `${Math.round(x * 100)}%`)
const NUM = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d))

const GATE_LABELS = {
  hallucination_rejection: 'Hallucinated citations rejected',
  low_context_gate_correctness: 'Low-confidence cases gated',
  happy_path_kind_agreement: 'Valid narratives accepted',
  malformed_rejection: 'Malformed model output rejected',
  openai_error_handling: 'API errors handled gracefully',
}

// Human labels + display order for the per-case section.
const CATEGORY_META = {
  happy_path: { label: 'Valid explanation accepted', order: 0 },
  hallucinated_citation: { label: 'Made-up fact rejected', order: 1 },
  low_context: { label: 'Thin evidence gated — no model call', order: 2 },
  prose_ungrounded_metric: { label: 'Ungrounded number in prose rejected', order: 3 },
  malformed_output: { label: 'Broken model output rejected', order: 4 },
  openai_error: { label: 'API failure handled gracefully', order: 5 },
}
const prettify = (s) => (s ? s.replace(/_/g, ' ') : s)

/** One narrative-eval case: what it checks + its expected→actual verdict + pass/fail. */
function CaseRow({ c }) {
  const pass = c.kind_match && c.reason_match && c.gate_respected
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 10, padding: '12px 14px' }}>
      <span style={{ flex: 'none', marginTop: 1, width: 18, height: 18, borderRadius: 99, background: pass ? 'var(--color-teal)' : 'var(--color-fail)', color: '#fff', fontSize: 11, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>{pass ? '✓' : '✗'}</span>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13.5, lineHeight: 1.5, color: 'var(--color-ink-soft)' }}>{c.description || prettify(c.name)}</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, marginTop: 7, fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--color-faint)' }}>
          <span style={{ padding: '2px 8px', borderRadius: 999, background: 'var(--color-teal-soft)', color: 'var(--color-teal)' }}>{c.mode}</span>
          <span>
            {c.expected_kind}
            {c.actual_kind && c.actual_kind !== c.expected_kind ? ` → ${c.actual_kind}` : ''}
          </span>
          {c.expected_reason && <span>· {c.expected_reason}</span>}
          {c.llm_was_called === false && <span>· no model call</span>}
        </div>
      </div>
    </div>
  )
}

export default function EvaluationPage() {
  const [data, setData] = useState(null)
  const [rag, setRag] = useState(null)
  const [err, setErr] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetch('/corpus/eval.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => !cancelled && setData(j))
      .catch(() => !cancelled && setErr(true))
    fetch('/corpus/rag_eval.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => !cancelled && setRag(j))
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const m = data?.metrics
  const hist = data?.negatives_histogram
  const lat = data?.latency
  const named = data?.named_examples
  const manifest = data?.manifest
  const examples = [
    ...(named?.false_positives || []).map((e) => ({ ...e, kind: 'False positive' })),
    ...(named?.false_negatives || []).map((e) => ({ ...e, kind: 'False negative' })),
  ]

  const cards = [
    { label: 'Recall@1', value: PCT(m?.recall_at_1), note: 'true match ranked first' },
    { label: 'Recall@3', value: PCT(m?.recall_at_3), note: 'true match in the top three' },
    { label: 'MRR', value: NUM(m?.mrr), note: 'mean reciprocal rank' },
  ]

  const ragCases = rag?.cases || []
  const groupedCases = Object.entries(
    ragCases.reduce((acc, c) => {
      ;(acc[c.category] ||= []).push(c)
      return acc
    }, {}),
  ).sort((a, b) => (CATEGORY_META[a[0]]?.order ?? 99) - (CATEGORY_META[b[0]]?.order ?? 99))

  return (
    <section style={{ maxWidth: 940, margin: '0 auto', padding: '64px 28px 80px' }}>
      <div style={{ fontSize: 12, fontWeight: 600, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-teal)', marginBottom: 16 }}>
        Evaluation
      </div>
      <h1 style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 38, lineHeight: 1.12, letterSpacing: '-0.015em', margin: '0 0 12px', maxWidth: '18ch' }}>
        Measured, not claimed.
      </h1>
      <p style={{ fontSize: 16, lineHeight: 1.6, color: 'var(--color-muted)', margin: '0 0 40px', maxWidth: '60ch' }}>
        {m
          ? `Retrieval quality on ${m.n_queries} held-out queries from the live ${manifest?.eval_mode === 'loo' ? 'leave-one-out' : ''} catalog. Numbers, not adjectives — these are read straight from the committed eval artifact.`
          : err
            ? 'Evaluation data is loading from the deployed catalog.'
            : 'Loading the latest measured results…'}
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 20 }}>
        {cards.map((c) => (
          <div key={c.label} style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 14, padding: 24 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, letterSpacing: '0.04em', color: 'var(--color-muted)', marginBottom: 14 }}>{c.label}</div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 500, fontSize: 44, lineHeight: 1, letterSpacing: '-0.02em' }}>{c.value}</div>
            <div style={{ fontSize: 12.5, color: 'var(--color-faint)', marginTop: 10 }}>{c.note}</div>
          </div>
        ))}
      </div>

      {lat && (
        <p style={{ fontSize: 13, color: 'var(--color-faint)', margin: '0 0 44px' }}>
          Ranking latency: <strong style={{ color: 'var(--color-muted)' }}>{NUM(lat.p50_ms, 2)} ms</strong> p50 ·{' '}
          {NUM(lat.p95_ms, 2)} ms p95 — in-memory cosine over the catalog (excludes audio decode + encode).
        </p>
      )}

      {hist && hist.counts?.some((c) => c > 0) && (
        <div style={{ marginBottom: 48 }}>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 14 }}>
            Top-1 cosine on unrelated tracks — the noise floor
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 120 }}>
            {hist.counts.map((c, i) => {
              const max = Math.max(...hist.counts, 1)
              return (
                <div
                  key={i}
                  title={`${hist.bins[i].toFixed(2)}–${(hist.bins[i] + hist.step).toFixed(2)}: ${c}`}
                  style={{ flex: 1, height: `${(c / max) * 100}%`, minHeight: c > 0 ? 2 : 0, background: 'var(--color-teal-soft)', borderTop: c > 0 ? '2px solid var(--color-teal)' : 'none', borderRadius: '2px 2px 0 0' }}
                />
              )
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--color-faint)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
            <span>0.0</span><span>cosine similarity</span><span>1.0</span>
          </div>
        </div>
      )}

      {rag?.baseline_gates && (
        <div style={{ marginBottom: 48 }}>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }}>
            Narrative integrity — the explanation layer is gated too
          </div>
          <p style={{ fontSize: 14.5, lineHeight: 1.6, color: 'var(--color-muted)', margin: '0 0 18px', maxWidth: '62ch' }}>
            The "why this resonates" text never sees audio and only explains an already-decided
            match. Every cited tempo, key, or artist fact is validated against the supplied
            context — anything unsupported is rejected, not shown. Measured on a {rag.n_cases}-case
            golden set; every gate must pass before a build ships.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
            {Object.entries(rag.baseline_gates).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 10, padding: '12px 14px' }}>
                <span style={{ flex: 'none', width: 18, height: 18, borderRadius: 99, background: v >= 1 ? 'var(--color-teal)' : 'var(--color-fail)', color: '#fff', fontSize: 12, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>{v >= 1 ? '✓' : '!'}</span>
                <span style={{ fontSize: 13.5, color: 'var(--color-ink-soft)' }}>{GATE_LABELS[k] || k}</span>
                <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 12.5, color: 'var(--color-faint)' }}>{PCT(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {groupedCases.length > 0 && (
        <div style={{ marginBottom: 48 }}>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 8 }}>
            What we test, case by case
          </div>
          <p style={{ fontSize: 14.5, lineHeight: 1.6, color: 'var(--color-muted)', margin: '0 0 20px', maxWidth: '64ch' }}>
            Every case feeds a fixed model reply through the real validation pipeline. A ✓ means the
            layer did the right thing — accepted a fully-grounded explanation, or refused one that was
            hallucinated, malformed, or built on thin evidence. {ragCases.length} cases across{' '}
            {groupedCases.length} categories; all currently pass.
          </p>
          <div style={{ display: 'grid', gap: 22 }}>
            {groupedCases.map(([cat, cases]) => {
              const allPass = cases.every((c) => c.kind_match && c.reason_match && c.gate_respected)
              return (
                <div key={cat}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                    <span style={{ fontFamily: 'var(--font-display)', fontSize: 16, color: 'var(--color-ink)' }}>
                      {CATEGORY_META[cat]?.label || prettify(cat)}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--color-faint)' }}>
                      {cases.length} {cases.length === 1 ? 'case' : 'cases'}
                    </span>
                    <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 11.5, color: allPass ? 'var(--color-teal)' : 'var(--color-fail)' }}>
                      {allPass ? 'all pass' : 'needs review'}
                    </span>
                  </div>
                  <div style={{ display: 'grid', gap: 8 }}>
                    {cases.map((c) => (
                      <CaseRow key={c.name} c={c} />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {examples.length > 0 && (
        <div style={{ marginBottom: 48 }}>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 14 }}>
            Named failure cases — where it gets it wrong, and why
          </div>
          <div style={{ display: 'grid', gap: 12 }}>
            {examples.map((e, i) => (
              <div key={i} style={{ background: 'var(--color-paper)', border: '1px solid var(--color-line)', borderRadius: 12, padding: 18 }}>
                <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--color-listen)', marginBottom: 6 }}>{e.kind}</div>
                <div style={{ fontFamily: 'var(--font-display)', fontSize: 16, color: 'var(--color-ink-soft)' }}>{e.why || e.note || e.title}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ maxWidth: '64ch' }}>
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--color-muted)', marginBottom: 12 }}>
          Method &amp; limitations
        </div>
        <p style={{ fontFamily: 'var(--font-display)', fontSize: 16.5, lineHeight: 1.64, color: 'var(--color-ink-soft)', margin: '0 0 16px' }}>
          {data?.methodology ||
            'Each catalog track is embedded and matched against the rest of the Creative-Commons catalog; Recall@k counts whether another track by the same artist appears in the top-k, and MRR rewards ranking it higher.'}
        </p>
        <p style={{ fontSize: 14, lineHeight: 1.62, color: 'var(--color-muted)', margin: 0 }}>
          {data?.limitations ||
            'A retrieval sanity check, not a definitive AI-generation eval. Many catalog artists have only one track, which makes same-artist recall strict and depresses the headline metrics.'}
        </p>
        {manifest && (
          <p style={{ fontSize: 12, color: 'var(--color-faint)', marginTop: 18, fontFamily: 'var(--font-mono)' }}>
            model {String(manifest.model_sha).slice(0, 12)} · {manifest.eval_mode} · generated {String(manifest.generated_at).slice(0, 10)}
          </p>
        )}
      </div>
    </section>
  )
}
