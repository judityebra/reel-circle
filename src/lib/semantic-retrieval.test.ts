import { describe, expect, it } from 'vitest'
import type { Movie, Recommendation } from './recommendation'
import { createSemanticVector, movieSemanticDocument, rerankWithSemanticSimilarity, semanticSearch } from './semantic-retrieval'

const movies: Movie[] = [
  {
    id: 'paris', title: 'Paris in October', year: 2020, runtime: 95, genres: ['Romance'], platforms: [], rating: 4,
    overview: 'A quiet rainy afternoon among warm cafes and falling leaves in France.', keywords: ['autumn', 'nostalgia'],
    director: 'A Director', cast: ['A Woman'], moods: ['Gentle'], language: 'French',
  },
  {
    id: 'space', title: 'Solar War', year: 2021, runtime: 95, genres: ['Sci-Fi'], platforms: [], rating: 4.8,
    overview: 'Pilots battle across a futuristic galaxy.', keywords: ['space battle'], moods: ['Electric'], language: 'English',
  },
]

const recommendations: Recommendation[] = movies.map((movie, index) => ({ movie, score: index === 0 ? 70 : 90, matchByProfile: [], reasons: [] }))

describe('semantic retrieval', () => {
  it('builds normalized deterministic vectors from rich movie documents', () => {
    const vector = createSemanticVector(movieSemanticDocument(movies[0]))
    const magnitude = Math.sqrt(vector.reduce((total, value) => total + value * value, 0))

    expect(vector).toHaveLength(384)
    expect(magnitude).toBeCloseTo(1)
  })

  it('retrieves concepts that do not map to a genre filter', () => {
    const [match] = semanticSearch('a rainy autumn Sunday in Paris', movies)

    expect(match.movieId).toBe('paris')
    expect(match.matchedTerms).toEqual(expect.arrayContaining(['rainy', 'autumn', 'paris']))
  })

  it('can rerank a stronger semantic match over a higher baseline score', () => {
    const ranked = rerankWithSemanticSimilarity('cozy rainy Paris autumn', recommendations)

    expect(ranked[0].movie.id).toBe('paris')
    expect(ranked[0].semanticScore).toBeGreaterThan(ranked[1].semanticScore)
  })
})
