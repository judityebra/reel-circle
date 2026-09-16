import { describe, expect, it } from 'vitest'
import { aggregateGroupScore, applyComparisonFeedback, applyTasteFeedback, chooseTasteQuestion, explainPreferenceMatch, scoreCandidates, type Movie, type Profile } from './recommendation'

const catalog: Movie[] = [
  { id: 'arrival-2016', title: 'Arrival', year: 2016, runtime: 116, genres: ['Drama', 'Sci-Fi'], platforms: ['Paramount+'], rating: 4.1 },
  { id: 'aftersun-2022', title: 'Aftersun', year: 2022, runtime: 102, genres: ['Drama'], platforms: ['MUBI'], rating: 4.2 },
  { id: 'clue-1985', title: 'Clue', year: 1985, runtime: 94, genres: ['Comedy', 'Mystery'], platforms: ['Prime Video'], rating: 3.8 },
]

const profiles: Profile[] = [
  {
    id: 'maya',
    name: 'Maya',
    color: '#ff6b35',
    ratings: [
      { title: 'Moonlight', year: 2016, rating: 5 },
      { title: 'Arrival', year: 2016, rating: 4.5 },
    ],
    watched: [{ title: 'Arrival', year: 2016 }],
    preferredGenres: ['Drama', 'Sci-Fi'],
  },
  {
    id: 'leo',
    name: 'Leo',
    color: '#11a683',
    ratings: [{ title: 'Knives Out', year: 2019, rating: 4.5 }],
    watched: [],
    preferredGenres: ['Comedy', 'Mystery'],
  },
]

describe('scoreCandidates', () => {
  it('excludes a film watched by anyone in the group', () => {
    const results = scoreCandidates(catalog, profiles, {})

    expect(results.map(({ movie }) => movie.title)).not.toContain('Arrival')
  })

  it('matches watched films by TMDB ID even when titles differ', () => {
    const localizedCatalog: Movie[] = [{ ...catalog[1], id: 'tmdb-42', tmdbId: 42, title: 'La hija oscura' }]
    const localizedProfiles: Profile[] = [{ ...profiles[0], watched: [{ title: 'The Lost Daughter', year: 2021, tmdbId: 42 }] }]

    expect(scoreCandidates(localizedCatalog, localizedProfiles, {})).toEqual([])
  })

  it('normalizes accents and punctuation for title/year fallback matching', () => {
    const normalizedCatalog: Movie[] = [{ ...catalog[1], id: 'amelie', title: 'Amélie', year: 2001 }]
    const normalizedProfiles: Profile[] = [{ ...profiles[0], watched: [{ title: 'Amelie!', year: 2001 }] }]

    expect(scoreCandidates(normalizedCatalog, normalizedProfiles, {})).toEqual([])
  })

  it('uses a runtime limit to choose a genuinely quick movie', () => {
    const results = scoreCandidates(catalog, profiles, { maxRuntime: 95 })

    expect(results[0].movie.title).toBe('Clue')
    expect(results.every(({ movie }) => movie.runtime <= 95)).toBe(true)
  })

  it('only returns movies on a selected streaming platform', () => {
    const results = scoreCandidates(catalog, profiles, { platforms: ['MUBI'] })

    expect(results.map(({ movie }) => movie.title)).toEqual(['Aftersun'])
  })

  it('enforces explicit genre exclusions', () => {
    const results = scoreCandidates(catalog, profiles, { excludedGenres: ['Drama'] })

    expect(results.map(({ movie }) => movie.title)).toEqual(['Clue'])
  })

  it('matches ISO language constraints to TMDB display languages', () => {
    const localizedCatalog = [
      { ...catalog[1], id: 'french', language: 'French' },
      { ...catalog[2], id: 'english', language: 'English' },
    ]

    expect(scoreCandidates(localizedCatalog, profiles, { languages: ['fr'] }).map(({ movie }) => movie.id)).toEqual(['french'])
  })

  it('uses learned reactions to adapt a person’s ranking', () => {
    const adaptableCatalog: Movie[] = [
      { id: 'drama', title: 'A Drama', year: 2020, runtime: 110, genres: ['Drama'], platforms: ['Filmin'], rating: 4 },
      { id: 'comedy', title: 'A Comedy', year: 2020, runtime: 110, genres: ['Comedy'], platforms: ['Filmin'], rating: 4 },
    ]
    const adaptableProfile: Profile = {
      ...profiles[0],
      watched: [],
      preferredGenres: [],
      tasteWeights: { Drama: -1, Comedy: 1 },
    }

    const results = scoreCandidates(adaptableCatalog, [adaptableProfile], {})

    expect(results[0].movie.title).toBe('A Comedy')
    expect(results.at(-1)?.movie.title).toBe('A Drama')
  })

  it('returns explainable per-person scores', () => {
    const [result] = scoreCandidates(catalog, profiles, { maxRuntime: 120 })

    expect(result.matchByProfile).toHaveLength(2)
    expect(result.reasons.length).toBeGreaterThan(0)
    expect(result.score).toBeGreaterThanOrEqual(0)
    expect(result.score).toBeLessThanOrEqual(100)
  })

  it('explains a recommendation from the active request', () => {
    expect(explainPreferenceMatch(catalog[2], { genres: ['Comedy'], maxRuntime: 100, excludedGenres: ['Horror'] })).toBe(
      'Clue matches your comedy preference, your under-100-minute limit. It also avoids horror.',
    )
  })

  it('blends neural predictions into explainable per-person scores', () => {
    const results = scoreCandidates(catalog, profiles, {
      neuralScores: { maya: { 'aftersun-2022': 100, 'clue-1985': 0 }, leo: { 'aftersun-2022': 100, 'clue-1985': 0 } },
      neuralBlend: 1,
    })

    expect(results[0].movie.title).toBe('Aftersun')
    expect(results[0].matchByProfile.every(({ score }) => score === 100)).toBe(true)
    expect(results[0].reasons).toContain('Neural taste model included')
  })
})

describe('applyTasteFeedback', () => {
  it('learns from skips without mutating the original profile', () => {
    const updated = applyTasteFeedback(profiles[0], catalog[1], 'skip')

    expect(updated.tasteWeights).toEqual({ Drama: -0.5 })
    expect(profiles[0].tasteWeights).toBeUndefined()
  })

  it('records a strong positive post-watch reaction', () => {
    const detailedMovie = { ...catalog[1], director: 'Charlotte Wells', language: 'English', moods: ['Tender'] }
    const updated = applyTasteFeedback(profiles[0], detailedMovie, 'love')

    expect(updated.tasteWeights).toEqual({ Drama: 0.75 })
    expect(updated.featureWeights).toEqual({
      'director:Charlotte Wells': 0.75,
      'language:English': 0.75,
      'mood:Tender': 0.75,
      'decade:2020s': 0.75,
    })
    expect(updated.watched).toContainEqual({ title: 'Aftersun', year: 2022 })
    expect(updated.ratings).toContainEqual({ title: 'Aftersun', year: 2022, rating: 5 })
  })
})

describe('active taste learning', () => {
  const questionCatalog: Movie[] = [
    { id: 'known-drama', title: 'Known Drama', year: 2020, runtime: 100, genres: ['Drama'], platforms: ['Filmin'], rating: 4 },
    { id: 'unknown-comedy', title: 'Unknown Comedy', year: 2021, runtime: 100, genres: ['Comedy'], platforms: ['Netflix'], rating: 4 },
    { id: 'unknown-mystery', title: 'Unknown Mystery', year: 2022, runtime: 100, genres: ['Mystery'], platforms: ['Movistar Plus+'], rating: 4 },
  ]

  it('asks about contrasting films in uncertain taste areas', () => {
    const profile = { ...profiles[0], watched: [], tasteWeights: { Drama: 2 } }

    const question = chooseTasteQuestion(questionCatalog, profile)

    expect(question?.map(({ id }) => id)).toEqual(['unknown-comedy', 'unknown-mystery'])
  })

  it('does not repeat movies that already have neural labels', () => {
    const question = chooseTasteQuestion(questionCatalog, { ...profiles[0], watched: [] }, new Set(['known-drama']))

    expect(question?.map(({ id }) => id)).toEqual(['unknown-comedy', 'unknown-mystery'])
  })

  it('learns from both the chosen and rejected film', () => {
    const profile = { ...profiles[0], watched: [], preferredGenres: [] }

    const updated = applyComparisonFeedback(profile, questionCatalog[1], questionCatalog[0])

    expect(updated.tasteWeights).toEqual({ Comedy: 0.25, Drama: -0.5 })
  })
})

describe('aggregateGroupScore', () => {
  it('supports average, least-misery, and Nash fairness strategies', () => {
    expect(aggregateGroupScore([100, 20], 'average')).toBe(60)
    expect(aggregateGroupScore([100, 20], 'least-misery')).toBe(20)
    expect(aggregateGroupScore([100, 20], 'nash')).toBe(45)
  })
})