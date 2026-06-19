import { motion } from 'framer-motion'
import { deriveHeadline } from '../lib/api.js'
import SimilarityReport from './SimilarityReport.jsx'
import AcrCloudRow from './AcrCloudRow.jsx'
import QualityBadge from './QualityBadge.jsx'

/**
 * ReportCard — the headline result. Composes:
 *   1. SimilarityReport      (Case A or Case B, depending on threshold)
 *   2. Two ACRCloud rows     (Cover Song ID + AI Music Detector)
 *   3. Quality badge         (inline + expandable 7-signal breakdown)
 *
 * Visual contract: ui_mockup_v2_suno_flare.html `article.report` block.
 *
 * Each section is separated by 1px line in --color-line; the ACRCloud signals
 * are independent rows that never compose into one verdict (LOCKED_DECISIONS Q10).
 *
 * @param {Object} props
 * @param {Object} props.neighbors - the /neighbors response payload
 * @param {Object} props.analyze   - the /analyze response (for the quality badge)
 * @param {boolean} [props.animate=true]
 */
export default function ReportCard({ neighbors, analyze, animate = true, queryFile = null }) {
  if (!neighbors) return null

  const {
    caseA,
    topPct,
    topLabel,
    topPercentile,
    topRawCosine,
    topSegment,
    querySpecificity,
  } = deriveHeadline(neighbors)
  const acrcloud = neighbors.acrcloud || {}
  const coverSongId = acrcloud.coverSongId || { status: 'disabled' }
  const aiMusicDetector = acrcloud.aiMusicDetector || { status: 'disabled' }

  return (
    <motion.article
      initial={animate ? { opacity: 0, y: 12 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
      className="p-10"
      style={{
        background: 'var(--color-bg)',
        border: '1px solid var(--color-line)',
        borderRadius: '4px',
      }}
    >
      <SimilarityReport
        caseA={caseA}
        neighbors={neighbors.neighbors}
        topPct={topPct}
        topLabel={topLabel}
        topPercentile={topPercentile}
        topRawCosine={topRawCosine}
        topSegment={topSegment}
        querySpecificity={querySpecificity}
        queryFile={queryFile}
        contextToken={neighbors.contextToken}
      />

      {/* ACRCloud signals — two independent rows, never composed into a verdict. */}
      <div className="mt-10">
        <AcrCloudRow variant="coverSongId" payload={coverSongId} status={coverSongId.status} />
        <AcrCloudRow variant="aiMusicDetector" payload={aiMusicDetector} status={aiMusicDetector.status} />
      </div>

      {/* Quality badge — inline + expandable. Consumes /analyze response. */}
      <div className="mt-6 border-t pt-6" style={{ borderColor: 'var(--color-line)' }}>
        <QualityBadge analyze={analyze} />
      </div>
    </motion.article>
  )
}
