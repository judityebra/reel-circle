import type { RecommendationOptions } from './recommendation'

const genreTerms: Array<[string, RegExp]> = [
  ['Comedy', /\b(comedy|comic|funny|comedia|divertid[oa]|gracios[oa]|romcom)\b/],
  ['Drama', /\b(drama|dramatic[oa])\b/],
  ['Mystery', /\b(mystery|misterio|intriga)\b/],
  ['Romance', /\b(romance|romantic|romantico|romantica|romcom)\b/],
  ['Sci-Fi', /\b(sci[ -]?fi|science fiction|ciencia ficcion)\b/],
  ['Horror', /\b(horror|scary|terror|miedo|spooky)\b/],
  ['Thriller', /\b(thriller|suspense|tension)\b/],
  ['Action', /\b(action|accion|adrenaline)\b/],
  ['Fantasy', /\b(fantasy|fantasia|magical|magica)\b/],
  ['Animation', /\b(animation|animated|animacion|anime)\b/],
  ['Documentary', /\b(documentary|documental)\b/],
  ['Crime', /\b(crime|criminal|crimen)\b/],
]

const moodTerms: Array<[string, RegExp]> = [
  ['Gentle', /\b(gentle|cozy|calm|suave|tranquil[oa]|acogedor[a]?)\b/],
  ['Playful', /\b(playful|witty|jugueton[a]?|ingenios[oa]|camp|campy|cunt|cunty|slay|girly|iconic)\b/],
  ['Electric', /\b(electric|intense|intens[oa]|energetic[oa])\b/],
  ['Reflective', /\b(reflective|thoughtful|reflexiv[oa]|contemplativ[oa])\b/],
  ['Twisty', /\b(twisty|mind.?bending|retorcid[oa]|sorprendente)\b/],
]

const platformTerms: Array<[string, RegExp]> = [
  ['Netflix', /\bnetflix\b/],
  ['Prime Video', /\b(prime video|amazon prime)\b/],
  ['Movistar Plus+', /\bmovistar(?: plus|\+)?\b/],
  ['Filmin', /\bfilmin\b/],
  ['MUBI', /\bmubi\b/],
  ['Max', /\b(?:hbo )?max\b/],
  ['Disney+', /\bdisney(?: plus|\+)?\b/],
  ['SkyShowtime', /\bskyshowtime\b/],
]

const normalize = (value: string) => value
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .toLowerCase()

export interface ConversationPreferences extends RecommendationOptions {
  excludedGenres: string[]
  peopleCount?: number
  viewingContext?: string
  lastUpdatedAt: number
}

const isNegated = (value: string, matchIndex: number) => {
  const prefix = value.slice(Math.max(0, matchIndex - 28), matchIndex)
  return /(?:\bno\b|\bnot\b|without|avoid|exclude|dont want|do not want|nada de|sin)\s*(?:anything\s+)?$/.test(prefix)
}

const extractGenreSignals = (normalized: string) => {
  const included: string[] = []
  const excluded: string[] = []
  genreTerms.forEach(([genre, pattern]) => {
    const match = pattern.exec(normalized)
    if (!match) return
    ;(isNegated(normalized, match.index) ? excluded : included).push(genre)
  })
  if (/\b(?:not futuristic|nothing futuristic|dont want anything futuristic|no futuristic)\b/.test(normalized)) excluded.push('Sci-Fi')
  return { included: [...new Set(included)], excluded: [...new Set(excluded)] }
}

export function createConversationPreferences(now = Date.now()): ConversationPreferences {
  return { genres: [], excludedGenres: [], platforms: [], moods: [], lastUpdatedAt: now }
}

export function updateConversationPreferences(
  current: ConversationPreferences,
  prompt: string,
  now = Date.now(),
): ConversationPreferences {
  const normalized = normalize(prompt).replace(/[’']/g, '')
  const reset = /\b(start over|reset|clear (?:my )?preferences|forget that|empezar de nuevo|borrar preferencias)\b/.test(normalized)
  const base = reset ? createConversationPreferences(now) : current
  const parsed = parseConstraintPrompt(prompt)
  const genreSignals = extractGenreSignals(normalized)
  const removeIntense = /\b(less intense|not (?:too )?intense|menos intens[oa])\b/.test(normalized)
  const peopleMatch = normalized.match(/\b(?:with|for|somos|con)\s+(\d{1,2})\s+(?:friends?|people|personas?|amigos?)\b/)
    ?? normalized.match(/\bmovie night with (\d{1,2})\b/)
  const yearMatch = normalized.match(/\b(?:from|after|since|desde|posterior a)\s+(19\d{2}|20\d{2})\b/)
  const decadeMatch = normalized.match(/\b((?:19|20)\d0)s\b/)
  const languageTerms: Array<[string, RegExp]> = [
    ['en', /\b(?:english|ingles)\b/],
    ['es', /\b(?:spanish|espanol)\b/],
    ['fr', /\b(?:french|frances)\b/],
    ['ko', /\b(?:korean|coreano)\b/],
    ['ja', /\b(?:japanese|japones)\b/],
  ]
  const languages = languageTerms.filter(([, pattern]) => pattern.test(normalized)).map(([language]) => language)
  const nextGenres = [...new Set([...(base.genres ?? []), ...genreSignals.included])]
    .filter((genre) => !genreSignals.excluded.includes(genre))
  const nextExcludedGenres = [...new Set([...base.excludedGenres, ...genreSignals.excluded])]
    .filter((genre) => !genreSignals.included.includes(genre))
  const nextMoods = [...new Set([...(base.moods ?? []), ...(parsed.moods ?? []), ...(/\b(dark|darker|mysterious|oscura?|misterios[oa])\b/.test(normalized) ? ['Twisty'] : [])])]
    .filter((mood) => !(removeIntense && mood === 'Electric'))

  return {
    ...base,
    genres: nextGenres,
    excludedGenres: nextExcludedGenres,
    platforms: parsed.platforms?.length ? parsed.platforms : base.platforms,
    moods: nextMoods,
    maxRuntime: parsed.maxRuntime ?? base.maxRuntime,
    rankingMode: parsed.rankingMode ?? base.rankingMode,
    yearRange: decadeMatch
      ? [Number(decadeMatch[1]), Number(decadeMatch[1]) + 9]
      : yearMatch ? [Number(yearMatch[1]), 2100] : base.yearRange,
    languages: languages.length ? languages : base.languages,
    peopleCount: peopleMatch ? Number(peopleMatch[1]) : base.peopleCount,
    viewingContext: /\b(date night|with my (?:boyfriend|girlfriend|partner)|cita)\b/.test(normalized)
      ? 'date-night'
      : /\b(movie night|noche de cine)\b/.test(normalized) ? 'movie-night' : base.viewingContext,
    lastUpdatedAt: now,
  }
}

export function parseConstraintPrompt(prompt: string): RecommendationOptions {
  const normalized = normalize(prompt)
  const genres = extractGenreSignals(normalized).included
  const platforms = platformTerms.filter(([, pattern]) => pattern.test(normalized)).map(([platform]) => platform)
  const moods = moodTerms.filter(([, pattern]) => pattern.test(normalized)).map(([mood]) => mood)
  const result: RecommendationOptions = { genres, platforms, moods }
  const hourMatch = normalized.match(/(?:under|less than|menos de|maximo)\s*(one|una?|two|dos|\d+(?:\.5)?)\s*(?:hours?|horas?)/)
  const hourValues: Record<string, number> = { one: 1, un: 1, una: 1, two: 2, dos: 2 }
  const runtimeMatch = normalized.match(/(?:under|less than|menos de|maximo)\s*(\d{2,3})/)

  if (hourMatch) result.maxRuntime = Math.round((hourValues[hourMatch[1]] ?? Number(hourMatch[1])) * 60)
  else if (/\b(hora y media|an hour and a half)\b/.test(normalized)) result.maxRuntime = 90
  else if (runtimeMatch) result.maxRuntime = Number(runtimeMatch[1])
  else if (/\b(short|quick|corta|rapida)\b/.test(normalized)) result.maxRuntime = 100

  if (/\b(fair|everyone|nadie odie|para todos)\b/.test(normalized)) {
    result.rankingMode = 'least-misery'
  } else if (/\b(consensus|balanced|equilibrad[oa])\b/.test(normalized)) {
    result.rankingMode = 'balanced'
  }

  return result
}

export function describeConstraints(options: RecommendationOptions): string[] {
  const descriptions = [
    ...(options.maxRuntime ? [`Under ${options.maxRuntime} min`] : []),
    ...(options.genres ?? []),
    ...(options.platforms ?? []),
    ...(options.moods ?? []),
    ...(options.rankingMode === 'least-misery' ? ['Nobody hates it'] : []),
    ...(options.rankingMode === 'balanced' ? ['Balanced group pick'] : []),
  ]
  return descriptions.length ? descriptions : ['Any movie']
}