import type { Movie } from './recommendation.js'

interface Provider {
  provider_name: string
}

export interface TmdbMovieDetails {
  id: number
  title: string
  release_date?: string
  runtime?: number
  vote_average?: number
  poster_path?: string | null
  backdrop_path?: string | null
  overview?: string
  genres?: Array<{ name: string }>
  original_language?: string
  credits?: { crew?: Array<{ job: string; name: string }>; cast?: Array<{ name: string; order?: number }> }
  keywords?: { keywords?: Array<{ name: string }> }
  videos?: { results?: Array<{ key: string; site: string; type: string; official?: boolean }> }
  providers?: { ES?: { flatrate?: Provider[]; free?: Provider[]; ads?: Provider[] } }
}

const languages: Record<string, string> = {
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  ja: 'Japanese',
  ko: 'Korean',
  de: 'German',
  it: 'Italian',
}

const inferMoods = (genres: string[]) => [
  ...(genres.includes('Comedy') ? ['Playful'] : []),
  ...(genres.includes('Mystery') || genres.includes('Thriller') ? ['Twisty'] : []),
  ...(genres.includes('Action') ? ['Electric'] : []),
  ...(genres.includes('Drama') ? ['Reflective'] : []),
]

const canonicalProvider = (provider: string) => {
  const aliases: Array<[RegExp, string]> = [
    [/amazon prime video/i, 'Prime Video'],
    [/movistar plus/i, 'Movistar Plus+'],
    [/hbo max/i, 'Max'],
    [/disney plus/i, 'Disney+'],
    [/netflix/i, 'Netflix'],
    [/skyshowtime/i, 'SkyShowtime'],
    [/\bmubi\b/i, 'MUBI'],
    [/\bfilmin\b/i, 'Filmin'],
  ]
  return aliases.find(([pattern]) => pattern.test(provider))?.[1] ?? provider
}

export function normalizeTmdbMovie(details: TmdbMovieDetails): Movie {
  const genres = details.genres?.map(({ name }) => name) ?? []
  const providerRegion = details.providers?.ES
  const platforms = [...new Set([
    ...(providerRegion?.flatrate ?? []),
    ...(providerRegion?.free ?? []),
    ...(providerRegion?.ads ?? []),
  ].map(({ provider_name }) => canonicalProvider(provider_name)))]
  const trailer = details.videos?.results?.find(({ site, type, official }) => site === 'YouTube' && type === 'Trailer' && official)
    ?? details.videos?.results?.find(({ site, type }) => site === 'YouTube' && type === 'Trailer')

  return {
    id: `tmdb-${details.id}`,
    tmdbId: details.id,
    title: details.title,
    year: Number(details.release_date?.slice(0, 4)) || 0,
    runtime: details.runtime ?? 0,
    genres,
    platforms,
    rating: Number(((details.vote_average ?? 0) / 2).toFixed(1)),
    poster: details.poster_path ? `https://image.tmdb.org/t/p/w500${details.poster_path}` : undefined,
    backdrop: details.backdrop_path ? `https://image.tmdb.org/t/p/w1280${details.backdrop_path}` : undefined,
    overview: details.overview?.trim() || undefined,
    keywords: details.keywords?.keywords?.map(({ name }) => name).slice(0, 20),
    cast: details.credits?.cast?.sort((left, right) => (left.order ?? 999) - (right.order ?? 999)).slice(0, 8).map(({ name }) => name),
    trailerUrl: trailer ? `https://www.youtube.com/watch?v=${trailer.key}` : undefined,
    director: details.credits?.crew?.find(({ job }) => job === 'Director')?.name,
    language: languages[details.original_language ?? ''] ?? details.original_language?.toUpperCase(),
    moods: inferMoods(genres),
    tone: inferMoods(genres)[0],
  }
}