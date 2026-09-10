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

export function parseConstraintPrompt(prompt: string): RecommendationOptions {
  const normalized = normalize(prompt)
  const genres = genreTerms.filter(([, pattern]) => pattern.test(normalized)).map(([genre]) => genre)
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