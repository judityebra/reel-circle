/** Mulberry32: small, fast, seedable PRNG returning floats in [0, 1). */
export function createRng(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Standard normal sample via Box-Muller. */
export function normal(rng: () => number, mean = 0, sd = 1): number {
  const u = Math.max(rng(), Number.EPSILON)
  const v = rng()
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

export function pick<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)]
}

export function shuffle<T>(rng: () => number, items: readonly T[]): T[] {
  const result = [...items]
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng() * (index + 1))
    ;[result[index], result[swap]] = [result[swap], result[index]]
  }
  return result
}

/**
 * Replace Math.random with a seeded generator for the lifetime of the process.
 * TensorFlow.js uses Math.random for batch shuffling, so this makes MLP training reproducible.
 * Returns a function that restores the original implementation.
 */
export function patchMathRandom(seed: number): () => void {
  const original = Math.random
  Math.random = createRng(seed)
  return () => {
    Math.random = original
  }
}
