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
  const rawQuery = Array.isArray(request.query?.q) ? request.query.q[0] : request.query?.q
  const query = rawQuery?.trim() ?? ''
  if (query.length < 2 || query.length > 80) return response.status(400).json({ error: 'Search query must be 2-80 characters' })

  try {
    const searchResponse = await fetch(tmdbUrl(`/search/movie?query=${encodeURIComponent(query)}&include_adult=false`, authentication), { headers: authentication.headers })
    if (!searchResponse.ok) throw new Error(`TMDB search failed: ${searchResponse.status}`)
    const search = await searchResponse.json() as { results?: Array<{ id: number }> }
    const movies = await Promise.all((search.results ?? []).slice(0, 10).map(({ id }) => fetchEnrichedMovie(id, authentication)))
    response.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=3600')
    return response.status(200).json({ movies: movies.filter(Boolean) })
  } catch {
    return response.status(502).json({ error: 'Movie search unavailable' })
  }
}