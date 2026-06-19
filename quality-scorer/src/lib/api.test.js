// Phase 3 unit tests — api.js `deriveHeadline` logic.
//
// Codex: when you wire up Vitest in Phase 3, this file should run green.
// The /neighbors integration is exercised end-to-end via the backend tests;
// this is just the pure JS threshold-rule logic that decides Case A vs Case B.

import { describe, it, expect } from 'vitest'
import { deriveHeadline } from './api.js'

describe('deriveHeadline', () => {
  const baseNeighbors = [
    {
      trackId: 'tier1:itunes:1',
      meanPooledSimilarity: 0.87,
      maxSegmentSimilarity: 0.92,
      track: { title: 'Blinding Lights', artist: 'The Weeknd' },
    },
    {
      trackId: 'tier1:itunes:2',
      meanPooledSimilarity: 0.72,
      maxSegmentSimilarity: 0.81,
      track: { title: 'Save Your Tears', artist: 'The Weeknd' },
    },
  ]

  it('returns Case A when top similarity is above threshold', () => {
    const out = deriveHeadline({
      neighbors: baseNeighbors,
      topMeanPooledSimilarity: 0.87,
      thresholdDefault: 0.70,
    })
    expect(out.caseA).toBe(true)
    expect(out.topPct).toBe(87)
    expect(out.topMatch.trackId).toBe('tier1:itunes:1')
  })

  it('returns Case B when top similarity is below threshold', () => {
    const out = deriveHeadline({
      neighbors: [
        {
          trackId: 'tier2:jamendo:42',
          meanPooledSimilarity: 0.55,
          maxSegmentSimilarity: 0.61,
          track: { title: 'Random Indie', artist: 'Some Artist' },
        },
      ],
      topMeanPooledSimilarity: 0.55,
      thresholdDefault: 0.70,
    })
    expect(out.caseA).toBe(false)
    expect(out.topPct).toBe(55)
    expect(out.topMatch.trackId).toBe('tier2:jamendo:42')
  })

  it('returns Case B with null top when neighbors is empty', () => {
    const out = deriveHeadline({
      neighbors: [],
      topMeanPooledSimilarity: 0,
      thresholdDefault: 0.70,
    })
    expect(out.caseA).toBe(false)
    expect(out.topPct).toBe(null)
    expect(out.topMatch).toBe(null)
  })

  it('handles no_corpus response gracefully', () => {
    const out = deriveHeadline({ verdict: 'no_corpus', neighbors: [] })
    expect(out.caseA).toBe(false)
    expect(out.topMatch).toBe(null)
  })

  it('formats the headline percentage to one decimal so close cosines stay distinguishable', () => {
    const out = deriveHeadline({
      neighbors: [
        { trackId: 'x', meanPooledSimilarity: 0.8734, track: {} },
      ],
      topMeanPooledSimilarity: 0.8734,
      thresholdDefault: 0.70,
    })
    expect(out.topPct).toBe(87.3)
  })

  it('distinguishes near-identical cosines at one-decimal precision', () => {
    const a = deriveHeadline({
      neighbors: [{ trackId: 'a', meanPooledSimilarity: 0.998, track: {} }],
      topMeanPooledSimilarity: 0.998,
      thresholdDefault: 0.70,
    })
    const b = deriveHeadline({
      neighbors: [{ trackId: 'b', meanPooledSimilarity: 0.996, track: {} }],
      topMeanPooledSimilarity: 0.996,
      thresholdDefault: 0.70,
    })
    expect(a.topPct).toBe(99.8)
    expect(b.topPct).toBe(99.6)
    expect(a.topPct).not.toBe(b.topPct)
  })

  it('uses 0.70 as default threshold when missing', () => {
    const out = deriveHeadline({
      neighbors: [{ trackId: 'x', meanPooledSimilarity: 0.71, track: {} }],
      topMeanPooledSimilarity: 0.71,
      // thresholdDefault omitted — should default to 0.70
    })
    expect(out.caseA).toBe(true)
  })

  it('honors a non-default threshold from the manifest', () => {
    const out = deriveHeadline({
      neighbors: [{ trackId: 'x', meanPooledSimilarity: 0.85, track: {} }],
      topMeanPooledSimilarity: 0.85,
      thresholdDefault: 0.90,
    })
    expect(out.caseA).toBe(false)
  })

  it('returns Case B when response is null/undefined', () => {
    expect(deriveHeadline(null).caseA).toBe(false)
    expect(deriveHeadline(undefined).caseA).toBe(false)
  })
})
