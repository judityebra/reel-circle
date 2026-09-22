import { afterEach, describe, expect, it } from 'vitest'
import { retrieveSemantically } from '../../api/semantic-service'
import type { Movie, Recommendation } from './recommendation'

const movies: Movie[] = [
  { id: 'paris', title: 'Paris in October', year: 2020, runtime: 95, genres: ['Romance'], platforms: [], rating: 4, overview: 'Rainy cafes and autumn leaves.' },
  { id: 'space', title: 'Solar War', year: 2021, runtime: 95, genres: ['Sci-Fi'], platforms: [], rating: 4.9, overview: 'A futuristic galaxy battle.' },
]
const recommendations: Recommendation[] = movies.map((movie, index) => ({ movie, score: index === 0 ? 70 : 90, matchByProfile: [], reasons: [] }))
const originalEnvironment = {
  embeddingBaseUrl: process.env.EMBEDDING_BASE_URL,
  embeddingModel: process.env.EMBEDDING_MODEL,
  qdrantUrl: process.env.QDRANT_URL,
}

afterEach(() => {
  const restore = (name: string, value: string | undefined) => {
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
  restore('EMBEDDING_BASE_URL', originalEnvironment.embeddingBaseUrl)
  restore('EMBEDDING_MODEL', originalEnvironment.embeddingModel)
  restore('QDRANT_URL', originalEnvironment.qdrantUrl)
})

describe('semantic service', () => {
  it('uses local semantic vectors with no external infrastructure', async () => {
    delete process.env.EMBEDDING_BASE_URL
    delete process.env.EMBEDDING_MODEL
    delete process.env.QDRANT_URL

    const result = await retrieveSemantically('rainy autumn Paris', recommendations)

    expect(result).toMatchObject({ retrievalMode: 'local', embeddingMode: 'local', embeddingModel: 'local-hash-v1', dimensions: 384 })
    expect(result.recommendations[0].movie.id).toBe('paris')
  })

  it('uses provider embeddings in memory when Qdrant is not configured', async () => {
    process.env.EMBEDDING_BASE_URL = 'https://embeddings.example/v1'
    process.env.EMBEDDING_MODEL = 'semantic-model'
    delete process.env.QDRANT_URL
    const embeddingFetcher = async () => ({
      ok: true,
      status: 200,
      json: async () => ({ data: [{ index: 0, embedding: [1, 0] }, { index: 1, embedding: [1, 0] }, { index: 2, embedding: [0, 1] }] }),
    })

    const result = await retrieveSemantically('something atmospheric', recommendations, { embeddingFetcher })

    expect(result).toMatchObject({ retrievalMode: 'embedding-memory', embeddingMode: 'provider', embeddingModel: 'semantic-model', dimensions: 2 })
    expect(result.recommendations[0].movie.id).toBe('paris')
  })

  it('falls back locally when the embedding provider fails', async () => {
    process.env.EMBEDDING_BASE_URL = 'https://embeddings.example/v1'
    process.env.EMBEDDING_MODEL = 'semantic-model'
    const embeddingFetcher = async () => ({ ok: false, status: 503, json: async () => ({}) })

    const result = await retrieveSemantically('rainy autumn Paris', recommendations, { embeddingFetcher })

    expect(result).toMatchObject({ retrievalMode: 'local', fallbackReason: 'embedding-failed' })
    expect(result.recommendations[0].movie.id).toBe('paris')
  })

  it('uses Qdrant results when both external services are configured', async () => {
    process.env.EMBEDDING_BASE_URL = 'https://embeddings.example/v1'
    process.env.EMBEDDING_MODEL = 'semantic-model'
    process.env.QDRANT_URL = 'https://qdrant.example'
    const embeddingFetcher = async () => ({
      ok: true,
      status: 200,
      json: async () => ({ data: [{ index: 0, embedding: [1, 0] }, { index: 1, embedding: [1, 0] }, { index: 2, embedding: [0, 1] }] }),
    })
    const qdrantResponses = [
      { ok: true, status: 200, json: async () => ({}) },
      { ok: true, status: 200, json: async () => ({}) },
      { ok: true, status: 200, json: async () => ({ result: [{ score: 0.95, payload: { movieId: 'paris' } }, { score: 0.05, payload: { movieId: 'space' } }] }) },
    ]
    const qdrantFetcher = async () => qdrantResponses.shift()!

    const result = await retrieveSemantically('atmospheric Paris', recommendations, { embeddingFetcher, qdrantFetcher })

    expect(result).toMatchObject({ retrievalMode: 'qdrant', embeddingMode: 'provider', embeddingModel: 'semantic-model', dimensions: 2 })
    expect(result.recommendations[0].movie.id).toBe('paris')
  })
})
