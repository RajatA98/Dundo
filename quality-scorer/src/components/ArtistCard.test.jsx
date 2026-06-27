import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import ArtistCard from './ArtistCard.jsx'

afterEach(cleanup)

const base = {
  artistId: 'jamendo:maya-lev',
  name: 'Maya Lev',
  similarity: 0.88,
  listenUrl: 'https://www.jamendo.com/artist/355362',
}

describe('ArtistCard — evidence overlap chips', () => {
  it('renders the shared-descriptor chips when evidenceTags is present', () => {
    render(
      <ArtistCard
        artist={{
          ...base,
          evidenceTags: {
            shared: [
              { kind: 'genre', label: 'rock', confidence: 0.6 },
              { kind: 'mood', label: 'atmospheric', confidence: 0.4 },
            ],
            confidence: 'high',
            method: 'mtg-knn-v1',
          },
        }}
      />,
    )
    expect(screen.getByText('You both lean')).toBeTruthy()
    expect(screen.getByText('rock')).toBeTruthy()
    expect(screen.getByText('atmospheric')).toBeTruthy()
  })

  it('renders no chip row when evidenceTags is absent (never padded)', () => {
    render(<ArtistCard artist={base} />)
    expect(screen.queryByText('You both lean')).toBeNull()
  })

  it('renders no chip row when shared is empty', () => {
    render(<ArtistCard artist={{ ...base, evidenceTags: { shared: [], confidence: 'low', method: 'mtg-knn-v1' } }} />)
    expect(screen.queryByText('You both lean')).toBeNull()
  })
})
