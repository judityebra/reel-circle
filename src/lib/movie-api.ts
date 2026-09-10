import type { Movie } from './recommendation'

interface CatalogResponse {
  movies?: Movie[]
  page?: number
  totalPages?: number
}

type Fetcher = (input: string) => Promise<Pick<Response, 'ok' | 'status' | 'json'>>

interface CatalogOptions {
  pages?: number
  startPage?: number
  onProgress?: (progress: { loadedPages: number; requestedPages: number; movies: number }) => void
}

export async function fetchMovieCatalog(
  baseUrl: string,
  fetcher: Fetcher = fetch,
  options: CatalogOptions = {},
): Promise<Movie[]> {
  const requestedPages = Math.max(1, Math.min(20, options.pages ?? 10))
  const startPage = Math.max(1, Math.min(500, options.startPage ?? 1))
  const moviesById = new Map<string, Movie>()

  for (let loadedPages = 1; loadedPages <= requestedPages; loadedPages += 1) {
    const page = startPage + loadedPages - 1
    const response = await fetcher(`${baseUrl.replace(/\/$/, '')}/catalog?region=ES&page=${page}`)
    if (!response.ok) {
      if (loadedPages === 1) throw new Error(`Movie service unavailable (${response.status})`)
      break
    }
    const payload = await response.json() as CatalogResponse
    if (!Array.isArray(payload.movies)) {
      if (loadedPages === 1) throw new Error('Movie service returned invalid data')
      break
    }
    payload.movies.forEach((movie) => moviesById.set(movie.id, movie))
    options.onProgress?.({ loadedPages, requestedPages, movies: moviesById.size })
    if (payload.totalPages !== undefined && page >= payload.totalPages) break
  }

  return [...moviesById.values()]
}

export async function searchMovieCatalog(
  baseUrl: string,
  query: string,
  fetcher: Fetcher = fetch,
): Promise<Movie[]> {
  const normalizedQuery = query.trim()
  if (normalizedQuery.length < 2) return []
  const response = await fetcher(`${baseUrl.replace(/\/$/, '')}/search?q=${encodeURIComponent(normalizedQuery)}&region=ES`)
  if (!response.ok) throw new Error(`Movie search unavailable (${response.status})`)
  const payload = await response.json() as CatalogResponse
  if (!Array.isArray(payload.movies)) throw new Error('Movie search returned invalid data')
  return payload.movies
}