import { fetchEnrichedMovie, getTmdbAuthentication, tmdbUrl } from './tmdb-client.js'

interface ServerlessRequest {
  method?: string
  query?: Record<string, string | string[] | undefined>
}

interface ServerlessResponse {
  status(code: number): ServerlessResponse
  json(body: unknown): void
  setHeader(name: string, value: string): void
}

export default async function handler(request: ServerlessRequest, response: ServerlessResponse) {
  if (request.method !== 'GET') return response.status(405).json({ error: 'Method not allowed' })
  const authentication = getTmdbAuthentication()
  if (!authentication) return response.status(503).json({ error: 'TMDB is not configured' })
  const requestedPage = Array.isArray(request.query?.page) ? request.query.page[0] : request.query?.page
  const page = Math.max(1, Math.min(500, Number(requestedPage) || 1))
  const discoverUrl = tmdbUrl(`/discover/movie?watch_region=ES&with_watch_monetization_types=flatrate%7Cfree%7Cads&sort_by=vote_average.desc&vote_count.gte=100&page=${page}`, authentication)

  try {
    const discoverResponse = await fetch(discoverUrl, { headers: authentication.headers })
    if (!discoverResponse.ok) throw new Error(`TMDB discover failed: ${discoverResponse.status}`)
    const discover = await discoverResponse.json() as { page?: number; total_pages?: number; results?: Array<{ id: number }> }
    const movies = await Promise.all((discover.results ?? []).map(({ id }) => fetchEnrichedMovie(id, authentication)))
    response.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate=86400')
    return response.status(200).json({
      page: discover.page ?? page,
      totalPages: discover.total_pages ?? page,
      movies: movies.filter(Boolean),
    })
  } catch {
    return response.status(502).json({ error: 'Movie service unavailable' })
  }
}