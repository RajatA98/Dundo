/**
 * Sample artist matches in the FROZEN artist contract shape
 * (backend/backend/artist.py :: ArtistMatch — artistId, name, similarity,
 * location?, listenUrl, supportLinks[], spotifyUrl?, narrative, criteria[]).
 *
 * This is the design's representative data, lifted from the approved
 * Dundo.dc.html. Phase 3 swaps this source for the real `/neighbors`
 * artist-framed response; the component contract stays identical.
 */
export const sampleArtists = [
  {
    artistId: 'jamendo:maya-lev',
    name: 'Maya Lev',
    location: 'Lisbon',
    similarity: 0.88,
    duration: '0:32',
    artGrad: 'linear-gradient(140deg,#0c8f86,#3A57D6)',
    narrative:
      'Both sit in a hushed F-minor, around 84 BPM, with a fingerpicked guitar floating over soft tape hiss. Maya leans more acoustic where yours adds a synth pad — but the patience, the room tone, and the way the melody hangs back are unmistakably shared.',
    listenUrl: 'https://mayalev.bandcamp.com',
    supportLinks: [
      { kind: 'bandcamp', label: 'Bandcamp', url: 'https://mayalev.bandcamp.com' },
      { kind: 'patreon', label: 'Ko-fi', url: 'https://ko-fi.com/mayalev' },
    ],
    spotifyUrl: 'https://open.spotify.com/artist/mayalev',
    criteria: [
      // Contract shape: { label, detail, agreement } — the card derives bar width + color.
      { label: 'Tempo', detail: '4 BPM apart', agreement: 0.92 },
      { label: 'Key', detail: 'Same key — F minor', agreement: 1.0 },
      { label: 'Harmonic', detail: 'Similar chord movement', agreement: 0.7 },
      { label: 'Timbre', detail: 'Warmer, more acoustic', agreement: 0.45 },
    ],
    // spectro is frontend presentation only (rendered from matched windows), NOT part of ArtistMatch.
    spectro: [
      { caption: 'your track · 0:42–0:52' },
      { caption: 'maya lev · 1:08–1:18' },
    ],
  },
  {
    artistId: 'fma:hollow-coast',
    name: 'Hollow Coast',
    location: 'Portland',
    similarity: 0.74,
    duration: '0:41',
    artGrad: 'linear-gradient(140deg,#3A57D6,#0c8f86)',
    narrative:
      'A wider, more reverberant take on the same idea — long sustained pads and a slow build that never quite resolves. Shares your track’s key center and that sense of drifting, but trades the close-mic intimacy for open space.',
    listenUrl: 'https://freemusicarchive.org/music/hollow-coast',
    supportLinks: [],
    spotifyUrl: 'https://open.spotify.com/artist/hollowcoast',
    criteria: [],
    spectro: [],
  },
  {
    artistId: 'jamendo:prata',
    name: 'Práta',
    location: null,
    similarity: 0.66,
    duration: '0:28',
    artGrad: 'linear-gradient(140deg,#0FB5A6,#1c8f86)',
    narrative:
      'Stripped to a single felt piano and breath, Práta finds the same melancholy you landed on. Faster and sparser, but the falling three-note figure at the heart of your track echoes here almost note-for-note.',
    listenUrl: 'https://jamendo.com/artist/prata',
    supportLinks: [],
    spotifyUrl: null,
    criteria: [],
    spectro: [],
  },
]

export const sampleMetrics = [
  { label: 'Recall@1', value: '0.71', note: 'true match ranked first' },
  { label: 'Recall@3', value: '0.89', note: 'true match in the top three' },
  { label: 'MRR', value: '0.81', note: 'mean reciprocal rank' },
]

/** Quiet similarity label from the cosine score (never a loud %). */
export function simLabel(similarity) {
  return similarity >= 0.85 ? 'strong resonance' : 'resonates with your track'
}
