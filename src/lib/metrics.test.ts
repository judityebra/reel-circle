import { describe, expect, it } from 'vitest'
import { summarizeFeedback } from './metrics'

describe('summarizeFeedback', () => {
  it('calculates acceptance and learning activity', () => {
    const metrics = summarizeFeedback([
      { id: '1', profileId: 'maya', movieId: 'a', action: 'skip', timestamp: 1 },
      { id: '2', profileId: 'maya', movieId: 'b', action: 'pick', timestamp: 2 },
      { id: '3', profileId: 'maya', movieId: 'c', action: 'love', timestamp: 3 },
      { id: '4', profileId: 'leo', movieId: 'd', action: 'comparison', timestamp: 4 },
    ])

    expect(metrics).toEqual({ interactions: 4, accepted: 2, skipped: 1, comparisons: 1, acceptanceRate: 50 })
  })
})