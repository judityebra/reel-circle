import { afterEach, describe, expect, it, vi } from 'vitest'
import { searchQdrant } from '../../api/qdrant-client'

const originalEnvironment = {
  url: process.env.QDRANT_URL,
  key: process.env.QDRANT_API_KEY,
  collection: process.env.QDRANT_COLLECTION,
}

afterEach(() => {
  vi.restoreAllMocks()
  const restore = (name: string, value: string | undefined) => {
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
  restore('QDRANT_URL', originalEnvironment.url)
  restore('QDRANT_API_KEY', originalEnvironment.key)
  restore('QDRANT_COLLECTION', originalEnvironment.collection)
})

describe('Qdrant client', () => {
  it('returns undefined when vector storage is not configured', async () => {
    delete process.env.QDRANT_URL

    await expect(searchQdrant([1, 0], [{ movieId: 'one', vector: [1, 0] }], 5)).resolves.toBeUndefined()
  })

  it('creates, upserts, and searches a candidate-scoped collection', async () => {
    process.env.QDRANT_URL = 'https://qdrant.example/'
    process.env.QDRANT_API_KEY = 'test-key'
    process.env.QDRANT_COLLECTION = 'movies'
    const fetcher = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ result: [{ score: 0.91, payload: { movieId: 'paris' } }] }) })

    const result = await searchQdrant([1, 0], [{ movieId: 'paris', vector: [1, 0] }, { movieId: 'space', vector: [0, 1] }], 2, fetcher)

    expect(result).toEqual([{ movieId: 'paris', score: 0.91, matchedTerms: [] }])
    expect(fetcher).toHaveBeenCalledTimes(4)
    expect(fetcher.mock.calls[0][0]).toBe('https://qdrant.example/collections/movies')
    expect(JSON.parse(fetcher.mock.calls[1][1].body).vectors).toEqual({ size: 2, distance: 'Cosine' })
    expect(JSON.parse(fetcher.mock.calls[3][1].body).filter.must[0].match.any).toEqual(['paris', 'space'])
    expect(fetcher.mock.calls[2][1].headers).toMatchObject({ 'api-key': 'test-key' })
  })
})
