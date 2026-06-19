import { useEffect, useMemo, useState } from 'react'

/**
 * Evaluation page — methodology + measured retrieval quality.
 *
 * Reads `/corpus/eval.json`. When present, renders the metric cards + histogram
 * against real numbers. When absent, the page still shows the same structure
 * — the methodology paragraph, the protocol explainer, the metric labels —
 * just with "not yet run" placeholders where the numbers would go.
 *
 * The page is built around the reframing in CODEX_PHASE_6_BACKEND_SCOPE_PROMPT:
 * the value isn't a definitive number, it's the demonstrated discipline of
 * having an eval + naming the limitations. That's the credibility signal for
 * a 5–20 min warm-intro read. Real numbers are a bonus.
 *
 * Backend produces eval.json via `python -m backend.scripts.run_eval --mode loo`.
 * Locked wire shape — see factory/artifacts/CODEX_PHASE_6.md.
 */
export default function EvaluationPage() {
  const [data, setData] = useState(null)
  const [missing, setMissing] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetch('/corpus/eval.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled) return
        if (!d || Object.keys(d).length === 0) setMissing(true)
        else setData(d)
      })
      .catch(() => { if (!cancelled) setMissing(true) })
    return () => { cancelled = true }
  }, [])

  const ready = !!data && !missing

  return (
    <div className="mx-auto max-w-[860px] py-16">
      <Kicker>measured, not claimed</Kicker>
      <h1
        className="mt-2"
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: '48px',
          fontWeight: 600,
          letterSpacing: '-0.025em',
          lineHeight: 1.08,
          color: 'var(--color-ink)',
        }}
      >
        Evaluation
      </h1>
      <p className="mt-4 text-[15px] leading-relaxed" style={{ color: 'var(--color-dim)' }}>
        Embedding-retrieval systems lie confidently when they&rsquo;re wrong. The only honest way
        to know whether this one works is to run a measured test, name the limitations, and
        publish the numbers. This page is that test.
      </p>

      <ProtocolSection />

      <MetricsSection data={data} ready={ready} />

      <LatencySection data={data} ready={ready} />

      <HistogramSection data={data} ready={ready} />

      <NamedExamplesSection data={data} ready={ready} />

      <ProseSection
        title="Methodology"
        body={data?.methodology ?? DEFAULT_METHODOLOGY}
      />

      <ProseSection
        title="Limitations"
        body={data?.limitations ?? DEFAULT_LIMITATIONS}
      />

      <ManifestFooter data={data} />
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Atoms                                                                   */
/* ----------------------------------------------------------------------- */

function Kicker({ children }) {
  return (
    <span
      className="block font-mono text-[12px] uppercase"
      style={{ color: 'var(--color-faint)', letterSpacing: '0.14em' }}
    >
      {children}
    </span>
  )
}

function SectionHeader({ kicker, title }) {
  return (
    <div className="mt-16">
      <Kicker>{kicker}</Kicker>
      <h2
        className="mt-2"
        style={{
          fontFamily: 'var(--font-display)',
          fontSize: '28px',
          fontWeight: 600,
          letterSpacing: '-0.02em',
          color: 'var(--color-ink)',
        }}
      >
        {title}
      </h2>
    </div>
  )
}

function ProseSection({ title, body }) {
  return (
    <section className="mt-12">
      <Kicker>{title}</Kicker>
      <p
        className="mt-2 text-[14px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        {body}
      </p>
    </section>
  )
}

/* ----------------------------------------------------------------------- */
/* Protocol section — always shown                                         */
/* ----------------------------------------------------------------------- */

function ProtocolSection() {
  return (
    <>
      <SectionHeader kicker="protocol" title="Retrieval check — leave-one-out over the catalog" />
      <p className="mt-4 text-[14px] leading-relaxed" style={{ color: 'var(--color-dim)' }}>
        For each track in the 160-track reference catalog, the track is held out of the index.
        The remaining 159 are searched using the held-out track&rsquo;s MuQ-MuLan embedding as the query.
        The held-out track&rsquo;s rank in the returned top-K is recorded along with the top-1 cosine
        similarity.
      </p>
      <p className="mt-3 text-[14px] leading-relaxed" style={{ color: 'var(--color-dim)' }}>
        This measures whether the embedding pipeline correctly retrieves the seed when given a
        held-out query — a <em style={{ color: 'var(--color-ink)' }}>catalog retrieval check</em>,
        not an end-to-end AI-soundalike test. The methodology and limitations sections at the
        bottom of this page name the trade-off explicitly.
      </p>

      {/* Metric explainer */}
      <div
        className="mt-6 grid gap-px"
        style={{
          gridTemplateColumns: '1fr 1fr 1fr',
          background: 'var(--color-line)',
          border: '1px solid var(--color-line)',
          borderRadius: '4px',
        }}
      >
        <MetricExplainer
          label="Recall@1"
          body="Share of queries whose held-out track lands at rank 1 in the returned neighbors. Strictest measure."
        />
        <MetricExplainer
          label="Recall@3"
          body="Share whose held-out track lands anywhere in the top 3. The UI returns 3 neighbors, so this is the user-visible accuracy."
        />
        <MetricExplainer
          label="MRR"
          body="Mean reciprocal rank of the held-out track across queries. Penalizes a #4 less than a #10; orders all queries on a continuous scale."
        />
      </div>
    </>
  )
}

function MetricExplainer({ label, body }) {
  return (
    <div className="p-5" style={{ background: 'var(--color-bg)' }}>
      <div
        className="font-mono text-[11px] uppercase"
        style={{ color: 'var(--color-faint)', letterSpacing: '0.08em' }}
      >
        {label}
      </div>
      <p
        className="mt-2 text-[13px] leading-relaxed"
        style={{ color: 'var(--color-dim)' }}
      >
        {body}
      </p>
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Metrics section — values from eval.json or placeholder                  */
/* ----------------------------------------------------------------------- */

function MetricsSection({ data, ready }) {
  const m = data?.metrics ?? {}
  return (
    <>
      <SectionHeader kicker="results" title={ready ? 'On the current corpus' : 'Numbers — not yet run'} />
      <div
        className="mt-6 grid"
        style={{
          gridTemplateColumns: 'repeat(3, 1fr)',
          border: '1px solid var(--color-line)',
          borderRadius: '4px',
        }}
      >
        <MetricCard label="Recall@1" value={fmt(m.recall_at_1)} n={m.n_queries} ready={ready} />
        <MetricCard label="Recall@3" value={fmt(m.recall_at_3)} n={m.n_queries} ready={ready} divider />
        <MetricCard label="MRR" value={fmt(m.mrr)} n={m.n_queries} ready={ready} divider />
      </div>
      {!ready && (
        <p
          className="mt-3 font-mono text-[12px]"
          style={{ color: 'var(--color-faint)' }}
        >
          run <code style={{ color: 'var(--color-ink)' }}>python -m backend.scripts.run_eval --mode loo</code> to populate.
        </p>
      )}
    </>
  )
}

function MetricCard({ label, value, n, ready, divider = false }) {
  return (
    <div
      className="p-6"
      style={{ borderLeft: divider ? '1px solid var(--color-line)' : 'none' }}
    >
      <div
        className="font-mono text-[12px] uppercase"
        style={{ color: 'var(--color-faint)', letterSpacing: '0.08em' }}
      >
        {label}
      </div>
      <div
        className="mt-1.5 tabular-nums"
        style={{
          fontSize: '48px',
          fontWeight: 700,
          letterSpacing: '-0.03em',
          color: ready ? 'var(--color-accent)' : 'var(--color-line)',
          lineHeight: 1,
        }}
      >
        {ready ? value : '—'}
      </div>
      {ready && Number.isFinite(n) && (
        <div
          className="mt-3 font-mono text-[12px]"
          style={{ color: 'var(--color-faint)' }}
        >
          n = {n}
        </div>
      )}
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Latency section — p50/p95/p99 wall-clock                                */
/* ----------------------------------------------------------------------- */

function LatencySection({ data, ready }) {
  const lat = data?.latency
  const have = ready && lat && Number.isFinite(lat.p50_ms)
  return (
    <>
      <SectionHeader
        kicker="latency"
        title={have ? 'How fast it runs' : 'Latency — not yet measured'}
      />
      <p className="mt-3 text-[14px] leading-relaxed" style={{ color: 'var(--color-dim)' }}>
        Wall-clock per <code style={{ color: 'var(--color-ink)' }}>/neighbors</code> ranking call
        against the in-memory catalog. Audio decode and MuQ-MuLan encoding aren&rsquo;t included — those
        are bounded by file size, not by index size. The rag-eval-harness methodology names latency
        as a first-class metric alongside precision and recall; slow systems get ignored.
      </p>
      <div
        className="mt-6 grid"
        style={{
          gridTemplateColumns: 'repeat(3, 1fr)',
          border: '1px solid var(--color-line)',
          borderRadius: '4px',
        }}
      >
        <LatencyCard label="p50" value={fmtMs(lat?.p50_ms)} ready={have} />
        <LatencyCard label="p95" value={fmtMs(lat?.p95_ms)} ready={have} divider />
        <LatencyCard label="p99" value={fmtMs(lat?.p99_ms)} ready={have} divider />
      </div>
      {have && Number.isFinite(lat.n_samples) && (
        <div className="mt-3 font-mono text-[12px]" style={{ color: 'var(--color-faint)' }}>
          n_samples = {lat.n_samples}
        </div>
      )}
    </>
  )
}

function LatencyCard({ label, value, ready, divider = false }) {
  return (
    <div
      className="p-6"
      style={{ borderLeft: divider ? '1px solid var(--color-line)' : 'none' }}
    >
      <div
        className="font-mono text-[12px] uppercase"
        style={{ color: 'var(--color-faint)', letterSpacing: '0.08em' }}
      >
        {label}
      </div>
      <div
        className="mt-1.5 tabular-nums"
        style={{
          fontSize: '40px',
          fontWeight: 700,
          letterSpacing: '-0.03em',
          color: ready ? 'var(--color-ink)' : 'var(--color-line)',
          lineHeight: 1,
        }}
      >
        {ready ? value : '—'}
      </div>
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Histogram section — top-1 cosine distribution                           */
/* ----------------------------------------------------------------------- */

function HistogramSection({ data, ready }) {
  const hist = data?.negatives_histogram
  const threshold = 0.70

  // Memo the bar layout so the SVG isn't recomputed on every render.
  const bars = useMemo(() => {
    if (!hist?.counts?.length) return null
    const max = Math.max(...hist.counts, 1)
    const step = hist.step ?? ((hist.bins?.[1] ?? 0.05) - (hist.bins?.[0] ?? 0))
    return hist.counts.map((c, i) => {
      const binStart = (hist.bins?.[i] ?? i * step) ?? 0
      return {
        i,
        height: (c / max) * 100,
        count: c,
        binStart,
        binEnd: binStart + step,
        overThreshold: binStart >= threshold,
      }
    })
  }, [hist])

  return (
    <>
      <SectionHeader
        kicker="noise floor"
        title={ready ? 'Where top-1 cosines land' : 'Score distribution — not yet run'}
      />
      <p className="mt-3 text-[14px] leading-relaxed" style={{ color: 'var(--color-dim)' }}>
        Histogram of top-1 cosine similarity across every LOO query. The vertical line marks the{' '}
        <code style={{ color: 'var(--color-ink)' }}>0.70</code> &ldquo;completely unique&rdquo; threshold — the
        cutoff below which the UI declares no match. The mass should cluster well above the line
        for a healthy retrieval pipeline (most catalog tracks should retrieve a near-self at rank 1);
        anything below the line is a retrieval miss and contributes to the noise floor.
      </p>

      <div
        className="mt-6 p-8"
        style={{
          border: '1px solid var(--color-line)',
          borderRadius: '4px',
          background: 'var(--color-bg)',
        }}
      >
        {ready && bars ? (
          <Histogram bars={bars} threshold={threshold} />
        ) : (
          <div className="flex h-[200px] items-center justify-center">
            <span
              className="font-mono text-[12px]"
              style={{ color: 'var(--color-faint)' }}
            >
              awaiting run_eval output
            </span>
          </div>
        )}
      </div>
    </>
  )
}

function Histogram({ bars, threshold }) {
  return (
    <div>
      {/* Bars */}
      <div
        className="relative flex items-end"
        style={{ height: '220px', gap: '3px', borderBottom: '1px solid var(--color-line)' }}
      >
        {bars.map((b) => (
          <div
            key={b.i}
            className="relative flex-1"
            style={{
              height: `${Math.max(b.height, 1)}%`,
              minHeight: '1px',
              background: b.overThreshold ? 'var(--color-accent)' : '#D4D5CB',
            }}
            title={`${b.binStart.toFixed(2)}–${b.binEnd.toFixed(2)}: ${b.count}`}
          />
        ))}
        {/* Threshold line at 0.70 */}
        <div
          className="absolute top-[-12px] bottom-0"
          style={{
            left: `${threshold * 100}%`,
            width: 0,
            borderLeft: '2px dashed var(--color-accent)',
            zIndex: 2,
          }}
        >
          <span
            className="absolute font-mono"
            style={{
              top: '-8px',
              left: '8px',
              fontSize: '11px',
              color: 'var(--color-accent)',
              whiteSpace: 'nowrap',
            }}
          >
            0.70 · completely-unique cutoff
          </span>
        </div>
      </div>
      {/* X axis */}
      <div
        className="mt-2 flex justify-between font-mono"
        style={{ fontSize: '11px', color: 'var(--color-faint)' }}
      >
        <span>0.0</span><span>0.2</span><span>0.4</span><span>0.6</span><span>0.8</span><span>1.0</span>
      </div>
      <div className="mt-1 flex justify-between font-mono" style={{ fontSize: '11px', color: 'var(--color-faint)' }}>
        <span>y · count of queries</span>
        <span>x · top-1 cosine similarity</span>
      </div>
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Named examples — placeholder until Option B activates                   */
/* ----------------------------------------------------------------------- */

function NamedExamplesSection({ data }) {
  const fps = data?.named_examples?.false_positives ?? []
  const fns = data?.named_examples?.false_negatives ?? []
  const anyCurated = fps.length > 0 || fns.length > 0

  return (
    <>
      <SectionHeader kicker="failure analysis" title="Where it's wrong, and why" />
      {anyCurated ? (
        <div className="mt-6 grid gap-8" style={{ gridTemplateColumns: '1fr 1fr' }}>
          <NamedColumn title="False positives" items={fps} />
          <NamedColumn title="False negatives" items={fns} />
        </div>
      ) : (
        <p
          className="mt-4 text-[14px] leading-relaxed"
          style={{ color: 'var(--color-dim)' }}
        >
          Curated examples — small set of hand-picked false positives and false negatives with
          audio playback and a one-sentence &ldquo;why this happened&rdquo; note — are queued as a follow-up
          pass when targeted Suno generations are available. The LOO numbers above already capture
          aggregate retrieval performance; the curated cards are the qualitative complement.
        </p>
      )}
    </>
  )
}

function NamedColumn({ title, items }) {
  return (
    <div>
      <Kicker>{title}</Kicker>
      <div className="mt-3 space-y-3">
        {items.map((ex, i) => (
          <NamedCard key={ex.id ?? i} item={ex} />
        ))}
      </div>
    </div>
  )
}

function NamedCard({ item }) {
  return (
    <div
      className="p-4"
      style={{
        border: '1px solid var(--color-line)',
        borderRadius: '4px',
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[15px] font-semibold" style={{ color: 'var(--color-ink)' }}>
          {item.query_title}
        </span>
        <span
          className="font-mono tabular-nums"
          style={{ fontSize: '13px', color: 'var(--color-ink)' }}
        >
          cos {Number(item.cosine).toFixed(2)}
        </span>
      </div>
      <div className="mt-1 text-[13px]" style={{ color: 'var(--color-dim)' }}>
        <span style={{ color: 'var(--color-faint)' }}>retrieved →</span> {item.retrieved_title}
      </div>
      {item.query_audio_url && (
        <audio controls preload="none" src={item.query_audio_url} className="mt-3 w-full" />
      )}
      {item.retrieved_audio_url && (
        <audio controls preload="none" src={item.retrieved_audio_url} className="mt-2 w-full" />
      )}
      {item.why && (
        <p
          className="mt-3 text-[13px] italic leading-relaxed"
          style={{ color: 'var(--color-dim)' }}
        >
          {item.why}
        </p>
      )}
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Manifest footer                                                          */
/* ----------------------------------------------------------------------- */

function ManifestFooter({ data }) {
  const m = data?.manifest
  if (!m) return null
  return (
    <div
      className="mt-16 border-t pt-6 font-mono"
      style={{
        borderColor: 'var(--color-line)',
        fontSize: '11px',
        color: 'var(--color-faint)',
        letterSpacing: '0.04em',
      }}
    >
      mode {m.mode ?? 'loo'} · model_sha {(m.model_sha ?? '').slice(0, 12)} ·
      generated {m.generated_at?.slice(0, 19) ?? '—'} ·
      n_queries {data?.metrics?.n_queries ?? '—'}
    </div>
  )
}

/* ----------------------------------------------------------------------- */
/* Helpers + defaults                                                       */
/* ----------------------------------------------------------------------- */

function fmt(v) {
  if (!Number.isFinite(v)) return '—'
  return v.toFixed(2)
}

function fmtMs(v) {
  if (!Number.isFinite(v)) return '—'
  if (v < 10) return `${v.toFixed(2)} ms`
  if (v < 100) return `${v.toFixed(1)} ms`
  return `${Math.round(v)} ms`
}

const DEFAULT_METHODOLOGY = `Leave-one-out retrieval check over the 160-track reference catalog. For each track, the track is held out of the index; the remaining 159 are queried using the held-out track's MuQ-MuLan embedding; the held-out track's rank in the returned top-K is recorded. This measures whether the embedding pipeline correctly finds the seed track when given a hold-out query — a catalog retrieval test, not an end-to-end AI-soundalike test. Because each query has at most one ground-truth target, Precision@1 equals Recall@1; we report Recall@k by convention. Groundedness (entity extraction from generated text) is not applicable — this system retrieves, it does not generate.`

const DEFAULT_LIMITATIONS = `Catalog is ~160 tracks, with Tier-1 from iTunes previews and Tier-2 from MTG-Jamendo. Tier-2 is anonymized in metadata, so per-artist or per-album similarity confounds may inflate Recall@K relative to a real-world deployment. The eval does NOT measure how the system handles AI-generated soundalikes; that requires Suno-targeted generations, queued as a follow-up curation pass.`
