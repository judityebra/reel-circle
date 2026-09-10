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
  genres?: Array<{ name: string }>
  original_language?: string
  credits?: { crew?: Array<{ job: string; name: string }> }
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
    director: details.credits?.crew?.find(({ job }) => job === 'Director')?.name,
    language: languages[details.original_language ?? ''] ?? details.original_language?.toUpperCase(),
    moods: inferMoods(genres),
    tone: inferMoods(genres)[0],
  }
}