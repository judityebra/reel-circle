import { describe, expect, it } from 'vitest'
import { compareRankings, explainRecommendationPath, getDisagreement } from './experiments'
import type { Movie, Profile, Recommendation } from './recommendation'

const movie: Movie = { id: 'one', title: 'One', year: 2022, runtime: 90, genres: ['Drama'], platforms: ['Filmin'], rating: 4, director: 'Jane Doe', language: 'Spanish', moods: ['Reflective'] }
const profile: Profile = { id: 'maya', name: 'Maya', color: '#fff', ratings: [], watched: [], preferredGenres: ['Drama'], tasteWeights: { Drama: 0.5 }, featureWeights: { 'director:Jane Doe': 1 } }
const recommendation = (id: string, scores: number[]): Recommendation => ({ movie: { ...movie, id, title: id }, score: 70, matchByProfile: scores.map((score, index) => ({ profileId: String(index), score })), reasons: [] })

describe('recommendation experiments', () => {
  it('measures top-pick changes and ranking overlap', () => {
    const result = compareRankings([recommendation('a', [70]), recommendation('b', [60]), recommendation('c', [50])], [recommendation('b', [80]), recommendation('a', [70]), recommendation('d', [60])])

    expect(result).toEqual({ baselineTop: 'a', modelTop: 'b', topChanged: true, topThreeOverlap: 67, meanRankShift: 1 })
  })

  it('calculates recommendation disagreement', () => {
    expect(getDisagreement(recommendation('a', [90, 30, 60]))).toBe(60)
  })

  it('returns the strongest inspectable taste path', () => {
    expect(explainRecommendationPath(profile, movie)).toEqual({
      label: 'Jane Doe',
      kind: 'director',
      weight: 1,
      path: ['Maya', 'director: Jane Doe', 'One'],
    })
  })
})