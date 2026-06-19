// Deterministic PRNG so the mock corpus is stable across reloads.
export function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)]
export const range = (rng, min, max) => min + rng() * (max - min)
export const chance = (rng, p) => rng() < p
