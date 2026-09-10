import { normalizeTmdbMovie, type TmdbMovieDetails } from '../src/lib/tmdb.js'

const tmdbBaseUrl = 'https://api.themoviedb.org/3'

export interface TmdbAuthentication {
  headers: Record<string, string>
  apiKey?: string
}

export function getTmdbAuthentication(): TmdbAuthentication | undefined {
  const credential = process.env.TMDB_BEARER_TOKEN ?? process.env.TMDB_API_KEY
  if (!credential) return undefined
  return credential.includes('.')
    ? { headers: { Authorization: `Bearer ${credential}`, Accept: 'application/json' } }
    : { headers: { Accept: 'application/json' }, apiKey: credential }
}

export function tmdbUrl(path: string, authentication: TmdbAuthentication): string {
  if (!authentication.apiKey) return `${tmdbBaseUrl}${path}`
  const separator = path.includes('?') ? '&' : '?'
  return `${tmdbBaseUrl}${path}${separator}api_key=${encodeURIComponent(authentication.apiKey)}`
}

export async function fetchEnrichedMovie(id: number, authentication: TmdbAuthentication) {
  const response = await fetch(tmdbUrl(`/movie/${id}?append_to_response=credits%2Cwatch%2Fproviders`, authentication), { headers: authentication.headers })
  if (!response.ok) return null
  const details = await response.json() as TmdbMovieDetails & {
    'watch/providers'?: { results?: TmdbMovieDetails['providers'] }
  }
  return normalizeTmdbMovie({ ...details, providers: details['watch/providers']?.results })
}