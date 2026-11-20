import { describe, expect, it } from 'vitest'
import { averageRanks, bootstrapCI, mean, ndcgAtK, precisionAtK, recall, sameSet, spearman } from './metrics'
import { createRng, patchMathRandom, shuffle } from './rng'

describe('ndcgAtK', () => {
  it('is 1 for the ideal ordering and lower for a reversed one', () => {
    const relevances = [5, 4, 3, 2, 1]
    expect(ndcgAtK(relevances, relevances, 3)).toBeCloseTo(1)
    expect(ndcgAtK([...relevances].reverse(), relevances, 3)).toBeLessThan(1)
  })

  it('matches a hand-computed value', () => {
    // DCG = 1/log2(2) + 3/log2(3) ; IDCG = 3/log2(2) + 1/log2(3)
    const expected = (1 + 3 / Math.log2(3)) / (3 + 1 / Math.log2(3))
    expect(ndcgAtK([1, 3], [3, 1], 2)).toBeCloseTo(expected)
  })

  it('returns 0 when nothing is relevant', () => {
    expect(ndcgAtK([0, 0], [0, 0], 10)).toBe(0)
  })
})

describe('precisionAtK', () => {
  it('counts items at or above the threshold in the top k', () => {
    expect(precisionAtK([5, 4, 3.5, 4.5, 1], 4, 4)).toBe(0.75)
  })

  it('uses the available items when fewer than k exist', () => {
    expect(precisionAtK([4, 2], 10, 4)).toBe(0.5)
    expect(precisionAtK([], 10, 4)).toBe(0)
  })
})

describe('spearman', () => {
  it('averages tied ranks', () => {
    expect(averageRanks([10, 20, 20, 30])).toEqual([1, 2.5, 2.5, 4])
  })

  it('is +1 for monotonic and -1 for reversed orderings', () => {
    expect(spearman([1, 2, 3, 4], [10, 20, 30, 40])).toBeCloseTo(1)
    expect(spearman([1, 2, 3, 4], [40, 30, 20, 10])).toBeCloseTo(-1)
  })

  it('scores a constant prediction as 0', () => {
    expect(spearman([50, 50, 50], [1, 2, 3])).toBe(0)
  })
})

describe('bootstrapCI', () => {
  it('brackets the sample mean and is reproducible for a seed', () => {
    const values = [0.2, 0.4, 0.6, 0.8, 1]
    const first = bootstrapCI(values, createRng(1))
    const second = bootstrapCI(values, createRng(1))
    expect(first).toEqual(second)
    expect(first.mean).toBeCloseTo(mean(values))
    expect(first.low).toBeLessThanOrEqual(first.mean)
    expect(first.high).toBeGreaterThanOrEqual(first.mean)
  })

  it('collapses for constant data', () => {
    expect(bootstrapCI([3, 3, 3], createRng(2))).toEqual({ mean: 3, low: 3, high: 3 })
  })
})

describe('set helpers', () => {
  it('compares sets ignoring order', () => {
    expect(sameSet(['a', 'b'], ['b', 'a'])).toBe(true)
    expect(sameSet(['a'], ['a', 'b'])).toBe(false)
    expect(sameSet(undefined, [])).toBe(true)
  })

  it('computes recall hits over expected items', () => {
    expect(recall(['Horror', 'Crime'], ['Horror'])).toEqual({ hits: 1, total: 2 })
  })
})

describe('seeded randomness', () => {
  it('produces the same sequence for the same seed', () => {
    expect(shuffle(createRng(7), [1, 2, 3, 4, 5])).toEqual(shuffle(createRng(7), [1, 2, 3, 4, 5]))
  })

  it('patches and restores Math.random', () => {
    const original = Math.random
    const restore = patchMathRandom(9)
    const first = Math.random()
    restore()
    const replay = patchMathRandom(9)
    expect(Math.random()).toBe(first)
    replay()
    expect(Math.random).toBe(original)
  })
})
