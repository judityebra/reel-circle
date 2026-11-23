import { describe, expect, it } from 'vitest'
import { createConversationPreferences, type ConversationPreferences } from '../src/lib/constraints'
import { extractionCases } from './extraction-cases'
import { fieldCorrect, normalizeViewingContext, scoreCase, summarize } from './extraction-scoring'

const state = (overrides: Partial<ConversationPreferences>): ConversationPreferences => ({ ...createConversationPreferences(0), ...overrides })

describe('fieldCorrect', () => {
  it('treats missing expectations as empty', () => {
    expect(fieldCorrect('genres', {}, state({}))).toBe(true)
    expect(fieldCorrect('genres', {}, state({ genres: ['Drama'] }))).toBe(false)
    expect(fieldCorrect('maxRuntime', {}, state({ maxRuntime: 90 }))).toBe(false)
  })

  it('compares lists as sets and normalises platform spelling', () => {
    expect(fieldCorrect('genres', { genres: ['Comedy', 'Romance'] }, state({ genres: ['Romance', 'Comedy'] }))).toBe(true)
    expect(fieldCorrect('platforms', { platforms: ['Max', 'Prime Video'] }, state({ platforms: ['HBO Max', 'prime video'] }))).toBe(true)
  })

  it('accepts any far-future end for open-ended year ranges', () => {
    expect(fieldCorrect('yearRange', { yearRange: [2010, null] }, state({ yearRange: [2010, 2100] }))).toBe(true)
    expect(fieldCorrect('yearRange', { yearRange: [2010, null] }, state({ yearRange: [2010, 2015] }))).toBe(false)
    expect(fieldCorrect('yearRange', { yearRange: [1990, 1999] }, state({ yearRange: [1990, 1999] }))).toBe(true)
    expect(fieldCorrect('yearRange', {}, state({ yearRange: [1990, 1999] }))).toBe(false)
  })

  it('folds free-text viewing contexts', () => {
    expect(normalizeViewingContext('Date night with partner')).toBe('date-night')
    expect(normalizeViewingContext('movie-night')).toBe('movie-night')
    expect(normalizeViewingContext(undefined)).toBeUndefined()
  })
})

describe('scoreCase', () => {
  it('flags an excluded genre that leaks into the requested genres', () => {
    const score = scoreCase({ excludedGenres: ['Horror'] }, state({ genres: ['Horror'] }))
    expect(score.contradiction).toBe(true)
    expect(score.exclusionHits).toBe(0)
    expect(score.exclusionTotal).toBe(1)
    expect(score.exactMatch).toBe(false)
  })

  it('counts a perfect prediction as an exact match', () => {
    const score = scoreCase({ genres: ['Comedy'], maxRuntime: 90 }, state({ genres: ['Comedy'], maxRuntime: 90 }))
    expect(score.exactMatch).toBe(true)
    expect(score.contradiction).toBe(false)
  })
})

describe('summarize', () => {
  it('aggregates field accuracy, exact match, exclusion recall and contradictions', () => {
    const summary = summarize([
      scoreCase({ excludedGenres: ['Horror', 'Drama'] }, state({ excludedGenres: ['Horror'] })),
      scoreCase({ genres: ['Comedy'] }, state({ genres: ['Comedy'] })),
    ])
    expect(summary.cases).toBe(2)
    expect(summary.exactMatch).toBe(0.5)
    expect(summary.exclusionRecall).toBe(0.5)
    expect(summary.contradictionRate).toBe(0)
    expect(summary.fieldAccuracy.excludedGenres).toBe(0.5)
    expect(summary.overallFieldAccuracy).toBeCloseTo(17 / 18)
  })
})

describe('extraction cases', () => {
  it('has about fifty unique bilingual cases including multi-turn ones', () => {
    const ids = extractionCases.map(({ id }) => id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(extractionCases.length).toBeGreaterThanOrEqual(45)
    expect(extractionCases.some(({ lang }) => lang === 'es')).toBe(true)
    expect(extractionCases.filter(({ turns }) => turns.length > 1).length).toBeGreaterThanOrEqual(8)
  })
})
