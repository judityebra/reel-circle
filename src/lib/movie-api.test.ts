import { describe, expect, it, vi } from 'vitest'
import { fetchMovieCatalog, searchMovieCatalog } from './movie-api'

describe('fetchMovieCatalog', () => {
  it('requests, deduplicates, and reports progress across Spain catalog pages', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ page: 1, totalPages: 3, movies: [{ id: 'one', title: 'One', year: 2020, runtime: 90, genres: [], platforms: [], rating: 4 }] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ page: 2, totalPages: 3, movies: [
          { id: 'one', title: 'One', year: 2020, runtime: 90, genres: [], platforms: [], rating: 4 },
          { id: 'two', title: 'Two', year: 2021, runtime: 95, genres: [], platforms: [], rating: 4 },
        ] }),
      })
    const onProgress = vi.fn()

    const movies = await fetchMovieCatalog('https://movies.example/api', fetcher, { pages: 2, onProgress })

    expect(fetcher).toHaveBeenNthCalledWith(1, 'https://movies.example/api/catalog?region=ES&page=1')
    expect(fetcher).toHaveBeenNthCalledWith(2, 'https://movies.example/api/catalog?region=ES&page=2')
    expect(movies.map(({ title }) => title)).toEqual(['One', 'Two'])
    expect(onProgress).toHaveBeenLastCalledWith({ loadedPages: 2, requestedPages: 2, movies: 2 })
  })

  it('fails safely when the backend is unavailable', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: false, status: 503 })

    await expect(fetchMovieCatalog('https://movies.example/api', fetcher, { pages: 1 })).rejects.toThrow('Movie service unavailable')
  })

  it('can continue loading from a later catalog page', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ page: 11, totalPages: 500, movies: [] }) })

    await fetchMovieCatalog('https://movies.example/api', fetcher, { pages: 1, startPage: 11 })

    expect(fetcher).toHaveBeenCalledWith('https://movies.example/api/catalog?region=ES&page=11')
  })

  it('searches titles through the server without exposing credentials', async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ movies: [{ id: 'searched', title: 'The Search', year: 2024, runtime: 100, genres: [], platforms: [], rating: 4 }] }),
    })

    const movies = await searchMovieCatalog('https://movies.example/api', 'The Search', fetcher)

    expect(fetcher).toHaveBeenCalledWith('https://movies.example/api/search?q=The%20Search&region=ES')
    expect(movies[0].id).toBe('searched')
  })
})