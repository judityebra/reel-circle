import type { Movie } from '../src/lib/recommendation'
import { normal, pick, shuffle } from './rng'

/**
 * Seeded synthetic world for offline recommendation benchmarks.
 *
 * Movies are generated from directors (each with a home genre and language), so genre,
 * director and language are correlated the way they are in real catalogs. Every movie has a
 * latent quality that drives its community rating. Users have latent preferences over genres,
 * moods, languages, directors, eras and runtime, plus a personal weight on quality, and a
 * hidden rating for every movie. Models only ever see the ratings sampled as labels.
 */

const GENRES = ['Action', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'] as const
const MOODS = ['Gentle', 'Playful', 'Electric', 'Reflective', 'Twisty'] as const
const LANGUAGES = ['English', 'Spanish', 'French', 'Korean', 'Japanese'] as const
const LANGUAGE_WEIGHTS = [0.5, 0.2, 0.12, 0.1, 0.08]
const PLATFORMS = ['Netflix', 'Prime Video', 'Filmin', 'MUBI', 'Max', 'Disney+', 'Movistar Plus+']

/** Moods that tend to accompany each genre; a movie draws most of its moods from here. */
const GENRE_MOODS: Record<string, string[]> = {
  Action: ['Electric'],
  Animation: ['Playful', 'Gentle'],
  Comedy: ['Playful'],
  Crime: ['Twisty', 'Electric'],
  Documentary: ['Reflective'],
  Drama: ['Reflective', 'Gentle'],
  Fantasy: ['Playful', 'Electric'],
  Horror: ['Electric', 'Twisty'],
  Mystery: ['Twisty'],
  Romance: ['Gentle'],
  'Sci-Fi': ['Twisty', 'Reflective'],
  Thriller: ['Electric', 'Twisty'],
}

export interface SyntheticDirector {
  name: string
  homeGenre: string
  homeLanguage: string
  talent: number
}

export interface SyntheticMovie extends Movie {
  quality: number
}

export interface SyntheticUser {
  id: string
  genreAffinity: Record<string, number>
  moodAffinity: Record<string, number>
  languageAffinity: Record<string, number>
  directorAffinity: Record<string, number>
  qualityWeight: number
  eraCenter: number
  runtimeComfort: number
  bias: number
  /** Hidden rating (0.5-5 in half stars) for every movie id. */
  ratings: Record<string, number>
}

export interface SyntheticWorld {
  directors: SyntheticDirector[]
  movies: SyntheticMovie[]
  users: SyntheticUser[]
}

export interface WorldOptions {
  movies: number
  users: number
  directors: number
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value))
const halfStar = (value: number) => Math.round(clamp(value, 0.5, 5) * 2) / 2

function weightedPick<T>(rng: () => number, items: readonly T[], weights: number[]): T {
  let threshold = rng() * weights.reduce((total, weight) => total + weight, 0)
  for (let index = 0; index < items.length; index += 1) {
    threshold -= weights[index]
    if (threshold <= 0) return items[index]
  }
  return items[items.length - 1]
}

function generateDirectors(rng: () => number, count: number): SyntheticDirector[] {
  return Array.from({ length: count }, (_, index) => ({
    name: `Director ${String(index + 1).padStart(2, '0')}`,
    homeGenre: GENRES[index % GENRES.length],
    homeLanguage: weightedPick(rng, LANGUAGES, LANGUAGE_WEIGHTS),
    talent: normal(rng, 0, 0.5),
  }))
}

function generateMovie(rng: () => number, index: number, director: SyntheticDirector): SyntheticMovie {
  const primary = rng() < 0.85 ? director.homeGenre : pick(rng, GENRES)
  const extraCount = rng() < 0.45 ? 0 : rng() < 0.75 ? 1 : 2
  const genres = [...new Set([primary, ...shuffle(rng, GENRES.filter((genre) => genre !== primary)).slice(0, extraCount)])]
  const moodPool = genres.flatMap((genre) => GENRE_MOODS[genre])
  const moods = [...new Set([pick(rng, moodPool), ...(rng() < 0.4 ? [rng() < 0.7 ? pick(rng, moodPool) : pick(rng, MOODS)] : [])])]
  const quality = normal(rng, 0, 0.8) + director.talent
  const year = 1960 + Math.floor(rng() * 65)
  const runtime = Math.round(clamp(normal(rng, 108, 20), 72, 195))
  const language = rng() < 0.8 ? director.homeLanguage : weightedPick(rng, LANGUAGES, LANGUAGE_WEIGHTS)
  const platforms = shuffle(rng, PLATFORMS).slice(0, 1 + Math.floor(rng() * 2))
  const id = `syn-${String(index + 1).padStart(3, '0')}`
  return {
    id,
    title: `Synthetic Film ${String(index + 1).padStart(3, '0')}`,
    year,
    runtime,
    genres,
    platforms,
    rating: Math.round(clamp(3.3 + 0.45 * quality + normal(rng, 0, 0.2), 1.5, 4.8) * 10) / 10,
    director: director.name,
    language,
    moods,
    quality,
  }
}

function generateUser(rng: () => number, index: number, directors: SyntheticDirector[]): Omit<SyntheticUser, 'ratings'> {
  const favourites = shuffle(rng, GENRES).slice(0, 2)
  const disliked = shuffle(rng, GENRES.filter((genre) => !favourites.includes(genre))).slice(0, 2)
  const genreAffinity = Object.fromEntries(GENRES.map((genre) => [
    genre,
    normal(rng, 0, 0.5) + (favourites.includes(genre) ? 1.4 : 0) - (disliked.includes(genre) ? 1.2 : 0),
  ]))
  return {
    id: `user-${String(index + 1).padStart(3, '0')}`,
    genreAffinity,
    moodAffinity: Object.fromEntries(MOODS.map((mood) => [mood, normal(rng, 0, 0.45)])),
    languageAffinity: Object.fromEntries(LANGUAGES.map((language) => [language, language === 'English' ? 0 : normal(rng, 0, 0.4)])),
    directorAffinity: Object.fromEntries(directors.map(({ name }) => [name, normal(rng, 0, 0.3)])),
    qualityWeight: 0.3 + rng() * 0.8,
    eraCenter: normal(rng, 1998, 14),
    runtimeComfort: 95 + rng() * 60,
    bias: normal(rng, 0, 0.3),
  }
}

/** Latent utility of a movie for a user, before noise. Exported for tests. */
function latentUtility(user: Omit<SyntheticUser, 'ratings'>, movie: SyntheticMovie): number {
  const genre = movie.genres.reduce((total, name) => total + user.genreAffinity[name], 0) / movie.genres.length
  const mood = (movie.moods ?? []).reduce((total, name) => total + user.moodAffinity[name], 0) / Math.max(1, movie.moods?.length ?? 0)
  const language = user.languageAffinity[movie.language ?? 'English'] ?? 0
  const director = user.directorAffinity[movie.director ?? ''] ?? 0
  const era = -Math.abs(movie.year - user.eraCenter) / 40
  const runtime = movie.runtime > user.runtimeComfort ? -(movie.runtime - user.runtimeComfort) / 40 : 0
  return user.bias + genre + mood + language + director + user.qualityWeight * movie.quality + era + runtime
}

export function generateWorld(rng: () => number, options: WorldOptions): SyntheticWorld {
  const directors = generateDirectors(rng, options.directors)
  const movies = Array.from({ length: options.movies }, (_, index) => generateMovie(rng, index, pick(rng, directors)))
  const users = Array.from({ length: options.users }, (_, index) => {
    const user = generateUser(rng, index, directors)
    const ratings = Object.fromEntries(movies.map((movie) => [
      movie.id,
      halfStar(3.7 + 0.95 * (latentUtility(user, movie) + normal(rng, 0, 0.45))),
    ]))
    return { ...user, ratings }
  })
  return { directors, movies, users }
}
