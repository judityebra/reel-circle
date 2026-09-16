export interface FilmRef {
  title: string
  year?: number
  tmdbId?: number
}

export interface Rating extends FilmRef {
  rating: number
}

export interface Movie extends FilmRef {
  id: string
  year: number
  runtime: number
  genres: string[]
  platforms: string[]
  rating: number
  poster?: string
  tone?: string
  director?: string
  language?: string
  moods?: string[]
}

export interface Profile {
  id: string
  name: string
  color: string
  ratings: Rating[]
  watched: FilmRef[]
  preferredGenres: string[]
  tasteWeights?: Record<string, number>
  featureWeights?: Record<string, number>
}

export type RankingMode = 'balanced' | 'average' | 'least-misery' | 'nash'

export interface RecommendationOptions {
  maxRuntime?: number
  genres?: string[]
  excludedGenres?: string[]
  platforms?: string[]
  moods?: string[]
  yearRange?: [number, number]
  languages?: string[]
  rankingMode?: RankingMode
  neuralScores?: Record<string, Record<string, number>>
  neuralBlend?: number
}

export interface Recommendation {
  movie: Movie
  score: number
  matchByProfile: Array<{ profileId: string; score: number }>
  reasons: string[]
}

export type TasteFeedback = 'skip' | 'pick' | 'love'
export function explainPreferenceMatch(movie: Movie, options: RecommendationOptions): string {
  const matchedGenres = (options.genres ?? []).filter((genre) => movie.genres.includes(genre))
  const matchedMoods = (options.moods ?? []).filter((mood) => movie.moods?.includes(mood))
  const details = [
    ...(matchedGenres.length ? [`your ${matchedGenres.join(' and ').toLowerCase()} preference`] : []),
    ...(matchedMoods.length ? [`the ${matchedMoods.join(' and ').toLowerCase()} mood`] : []),
    ...(options.maxRuntime ? [`your under-${options.maxRuntime}-minute limit`] : []),
    ...(options.yearRange ? [`your ${options.yearRange[0]}-${options.yearRange[1]} release window`] : []),
    ...(options.languages?.length && movie.language ? [`your ${movie.language.toUpperCase()} language choice`] : []),
  ]
  if (!details.length) return `${movie.title} rises to the top from the group's ratings and learned taste signals.`
  const exclusions = options.excludedGenres?.length ? ` It also avoids ${options.excludedGenres.join(' and ').toLowerCase()}.` : ''
  return `${movie.title} matches ${details.join(', ')}.${exclusions}`
}

const normalizedTitle = (title: string) => title
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .toLocaleLowerCase()
  .replace(/[^a-z0-9]+/g, ' ')
  .trim()

const filmKey = ({ title, year, tmdbId }: FilmRef) =>
  tmdbId ? `tmdb:${tmdbId}` : `${normalizedTitle(title)}::${year ?? ''}`

const clamp = (value: number) => Math.max(0, Math.min(100, Math.round(value)))

const languageAliases: Record<string, string> = {
  en: 'english',
  es: 'spanish',
  fr: 'french',
  ja: 'japanese',
  ko: 'korean',
  de: 'german',
  it: 'italian',
}

const normalizedLanguage = (value: string) => languageAliases[value.toLowerCase()] ?? value.toLowerCase()

export const getMovieFeatures = (movie: Movie) => [
  ...(movie.director ? [`director:${movie.director}`] : []),
  ...(movie.language ? [`language:${movie.language}`] : []),
  ...(movie.moods ?? []).map((mood) => `mood:${mood}`),
  `decade:${Math.floor(movie.year / 10) * 10}s`,
]

export function aggregateGroupScore(scores: number[], mode: RankingMode): number {
  if (scores.length === 0) return 0
  if (mode === 'least-misery') return clamp(Math.min(...scores))
  if (mode === 'nash') {
    const product = scores.reduce((total, score) => total * Math.max(score / 100, 0.01), 1)
    return clamp(Math.pow(product, 1 / scores.length) * 100)
  }

  const average = scores.reduce((sum, score) => sum + score, 0) / scores.length
  if (mode === 'average') return clamp(average)
  const disagreement = scores.length > 1 ? Math.max(...scores) - Math.min(...scores) : 0
  return clamp(average - disagreement * 0.2)
}

export function applyTasteFeedback(
  profile: Profile,
  movie: Movie,
  feedback: TasteFeedback,
): Profile {
  const delta = feedback === 'skip' ? -0.5 : feedback === 'pick' ? 0.25 : 0.75
  const tasteWeights = { ...profile.tasteWeights }
  const featureWeights = { ...profile.featureWeights }
  movie.genres.forEach((genre) => {
    tasteWeights[genre] = Math.max(-2, Math.min(2, (tasteWeights[genre] ?? 0) + delta))
  })
  getMovieFeatures(movie).forEach((feature) => {
    featureWeights[feature] = Math.max(-2, Math.min(2, (featureWeights[feature] ?? 0) + delta))
  })

  if (feedback !== 'love') {
    return { ...profile, tasteWeights, featureWeights }
  }

  const watched = profile.watched.some((film) => filmKey(film) === filmKey(movie))
    ? profile.watched
    : [...profile.watched, { title: movie.title, year: movie.year, tmdbId: movie.tmdbId }]
  const ratings = [
    ...profile.ratings.filter((rating) => filmKey(rating) !== filmKey(movie)),
    { title: movie.title, year: movie.year, tmdbId: movie.tmdbId, rating: 5 },
  ]

  return { ...profile, tasteWeights, featureWeights, watched, ratings }
}

export function applyComparisonFeedback(
  profile: Profile,
  chosen: Movie,
  rejected: Movie,
): Profile {
  return applyTasteFeedback(applyTasteFeedback(profile, chosen, 'pick'), rejected, 'skip')
}

export function chooseTasteQuestion(
  movies: Movie[],
  profile: Profile,
  excludedMovieIds: ReadonlySet<string> = new Set(),
): [Movie, Movie] | undefined {
  const watched = new Set(profile.watched.map(filmKey))
  const candidates = movies.filter((movie) => !watched.has(filmKey(movie)) && !excludedMovieIds.has(movie.id))
  if (candidates.length < 2) return undefined

  let bestPair: [Movie, Movie] = [candidates[0], candidates[1]]
  let bestInformationScore = Number.NEGATIVE_INFINITY

  for (let leftIndex = 0; leftIndex < candidates.length - 1; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
      const left = candidates[leftIndex]
      const right = candidates[rightIndex]
      const sharedGenres = left.genres.filter((genre) => right.genres.includes(genre)).length
      const genreContrast = new Set([...left.genres, ...right.genres]).size - sharedGenres
      const uncertainty = [...left.genres, ...right.genres].reduce(
        (total, genre) => total + Math.abs(profile.tasteWeights?.[genre] ?? 0),
        0,
      )
      const informationScore = genreContrast * 10 - uncertainty * 4 + (left.rating + right.rating) / 2

      if (informationScore > bestInformationScore) {
        bestPair = [left, right]
        bestInformationScore = informationScore
      }
    }
  }

  return bestPair
}

export function scoreCandidates(
  catalog: Movie[],
  profiles: Profile[],
  options: RecommendationOptions,
): Recommendation[] {
  const watched = new Set(profiles.flatMap((profile) => profile.watched.map(filmKey)))
  const requestedGenres = options.genres ?? []
  const excludedGenres = options.excludedGenres ?? []
  const requestedPlatforms = options.platforms ?? []
  const requestedMoods = options.moods ?? []

  return catalog
    .filter((movie) => !watched.has(filmKey(movie)))
    .filter((movie) => !options.maxRuntime || movie.runtime <= options.maxRuntime)
    .filter((movie) => !excludedGenres.some((genre) => movie.genres.includes(genre)))
    .filter((movie) => !options.yearRange || (movie.year >= options.yearRange[0] && movie.year <= options.yearRange[1]))
    .filter((movie) => !options.languages?.length || (movie.language && options.languages.some((language) => normalizedLanguage(language) === normalizedLanguage(movie.language!))))
    .filter(
      (movie) =>
        requestedGenres.length === 0 ||
        requestedGenres.some((genre) => movie.genres.includes(genre)),
    )
    .filter(
      (movie) =>
        requestedPlatforms.length === 0 ||
        requestedPlatforms.some((platform) => movie.platforms.includes(platform)),
    )
    .filter(
      (movie) =>
        requestedMoods.length === 0 ||
        requestedMoods.some((mood) => movie.moods?.includes(mood)),
    )
    .map((movie) => {
      const matchByProfile = profiles.map((profile) => {
        const genreMatches = movie.genres.filter((genre) =>
          profile.preferredGenres.includes(genre),
        ).length
        const learnedPreference = movie.genres.reduce(
          (total, genre) => total + (profile.tasteWeights?.[genre] ?? 0),
          0,
        )
        const featurePreference = getMovieFeatures(movie).reduce(
          (total, feature) => total + (profile.featureWeights?.[feature] ?? 0),
          0,
        )
        const explainableScore = clamp(movie.rating * 14 + genreMatches * 14 + learnedPreference * 16 + featurePreference * 6 + (movie.runtime <= 100 ? 8 : 0))
        const neuralScore = options.neuralScores?.[profile.id]?.[movie.id]
        const blend = Math.max(0, Math.min(1, options.neuralBlend ?? 0))
        const score = neuralScore === undefined ? explainableScore : clamp(explainableScore * (1 - blend) + neuralScore * blend)
        return { profileId: profile.id, score }
      })
      const scores = matchByProfile.map(({ score }) => score)
      const score = aggregateGroupScore(scores, options.rankingMode ?? 'balanced')
      const sharedGenres = movie.genres.filter((genre) =>
        profiles.every((profile) => profile.preferredGenres.includes(genre)),
      )
      const reasons = [
        sharedGenres.length > 0
          ? `Shared taste: ${sharedGenres.join(', ')}`
          : `${movie.genres[0]} bridges this group`,
        movie.runtime <= 100 ? `Quick ${movie.runtime}-minute watch` : `${movie.runtime} minutes`,
        `${movie.rating.toFixed(1)} community rating`,
        ...(options.neuralScores && options.neuralBlend ? ['Neural taste model included'] : []),
      ]

      return { movie, score, matchByProfile, reasons }
    })
    .sort((left, right) => right.score - left.score)
}