/**
 * AcrCloudRow rendering tests — lock the three AI Music Detector verdict
 * branches and the Cover Song ID match/no-match branches so the wire-shape
 * contract from LOCKED_DECISIONS Q10 stays faithful in the UI.
 */

import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import AcrCloudRow from './AcrCloudRow.jsx'

afterEach(cleanup)

describe('AcrCloudRow — Cover Song ID branch', () => {
  it('renders the match copy with title + artist + score', () => {
    render(
      <AcrCloudRow
        variant="coverSongId"
        status="match"
        payload={{
          status: 'match',
          title: 'Blinding Lights',
          artist: 'The Weeknd',
          score: 88,
        }}
      />,
    )
    expect(screen.getByText(/ACRCLOUD · COVER SONG ID/)).toBeInTheDocument()
    expect(screen.getByText(/Cover match:/)).toBeInTheDocument()
    expect(screen.getByText(/Blinding Lights/)).toBeInTheDocument()
    expect(screen.getByText(/The Weeknd/)).toBeInTheDocument()
    expect(screen.getByText(/88%/)).toBeInTheDocument()
  })

  it('renders "No cover match" when status is no_match', () => {
    render(<AcrCloudRow variant="coverSongId" status="no_match" payload={null} />)
    expect(screen.getByText(/No cover match/)).toBeInTheDocument()
  })

  it('renders the disabled copy when status is disabled', () => {
    render(<AcrCloudRow variant="coverSongId" status="disabled" payload={null} />)
    expect(
      screen.getByText(/Signal unavailable in public demo/),
    ).toBeInTheDocument()
  })
})

describe('AcrCloudRow — AI Music Detector verdict dispatch', () => {
  it('verdict=ai_generated + likely_source=suno renders the Suno rose pill', () => {
    render(
      <AcrCloudRow
        variant="aiMusicDetector"
        status="match"
        payload={{
          status: 'match',
          verdict: 'ai_generated',
          ai_probability: 87,
          likely_source: 'suno',
        }}
      />,
    )
    expect(screen.getByText(/AI-generated/)).toBeInTheDocument()
    expect(screen.getByText(/87%/)).toBeInTheDocument()
    // Suno pill text ("likely suno") — confirms the brand-aware moment.
    expect(screen.getByText(/likely suno/i)).toBeInTheDocument()
  })

  it('verdict=ai_generated + likely_source=udio renders plain "likely udio"', () => {
    render(
      <AcrCloudRow
        variant="aiMusicDetector"
        status="match"
        payload={{
          status: 'match',
          verdict: 'ai_generated',
          ai_probability: 91,
          likely_source: 'udio',
        }}
      />,
    )
    expect(screen.getByText(/AI-generated/)).toBeInTheDocument()
    expect(screen.getByText(/91%/)).toBeInTheDocument()
    expect(screen.getByText(/likely udio/)).toBeInTheDocument()
  })

  it('verdict=human renders "Likely human (5% AI probability)" — not "AI-generated"', () => {
    render(
      <AcrCloudRow
        variant="aiMusicDetector"
        status="match"
        payload={{
          status: 'match',
          verdict: 'human',
          ai_probability: 5,
          likely_source: null,
        }}
      />,
    )
    // This is the regression Codex caught — must not say "AI-generated".
    expect(screen.queryByText(/AI-generated/)).not.toBeInTheDocument()
    expect(screen.queryByText(/likely unknown/i)).not.toBeInTheDocument()
    expect(screen.getByText(/Likely human/)).toBeInTheDocument()
    expect(screen.getByText(/5% AI probability/)).toBeInTheDocument()
  })

  it('verdict=no_vocals renders "No vocals detected"', () => {
    render(
      <AcrCloudRow
        variant="aiMusicDetector"
        status="match"
        payload={{
          status: 'match',
          verdict: 'no_vocals',
          ai_probability: 0,
          likely_source: null,
        }}
      />,
    )
    expect(screen.getByText(/No vocals detected/)).toBeInTheDocument()
    expect(screen.queryByText(/AI-generated/)).not.toBeInTheDocument()
  })

  it('ai_generated with no likely_source renders "source unclear" instead of "likely unknown"', () => {
    render(
      <AcrCloudRow
        variant="aiMusicDetector"
        status="match"
        payload={{
          status: 'match',
          verdict: 'ai_generated',
          ai_probability: 72,
          likely_source: null,
        }}
      />,
    )
    expect(screen.getByText(/AI-generated/)).toBeInTheDocument()
    expect(screen.getByText(/72%/)).toBeInTheDocument()
    expect(screen.getByText(/source unclear/)).toBeInTheDocument()
    expect(screen.queryByText(/likely unknown/i)).not.toBeInTheDocument()
  })

  it('clamps an out-of-range ai_probability to [0,100]', () => {
    render(
      <AcrCloudRow
        variant="aiMusicDetector"
        status="match"
        payload={{
          status: 'match',
          verdict: 'ai_generated',
          ai_probability: 175,
          likely_source: 'suno',
        }}
      />,
    )
    expect(screen.getByText(/100%/)).toBeInTheDocument()
  })
})

describe('AcrCloudRow — degraded states', () => {
  it('status=timeout renders the unavailable copy', () => {
    render(<AcrCloudRow variant="aiMusicDetector" status="timeout" payload={null} />)
    expect(
      screen.getByText(/Second-opinion service unavailable/),
    ).toBeInTheDocument()
  })

  it('status=quota_exceeded renders the unavailable copy', () => {
    render(<AcrCloudRow variant="coverSongId" status="quota_exceeded" payload={null} />)
    expect(
      screen.getByText(/Second-opinion service unavailable/),
    ).toBeInTheDocument()
  })
})
