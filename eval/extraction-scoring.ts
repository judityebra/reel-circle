import type { ConversationPreferences } from '../src/lib/constraints'
import type { ExpectedPreferences } from './extraction-cases'
import { sameSet } from './metrics'

export const FIELDS = ['genres', 'excludedGenres', 'moods', 'platforms', 'maxRuntime', 'yearRange', 'languages', 'peopleCount', 'viewingContext'] as const
export type Field = (typeof FIELDS)[number]

/** Year at or beyond which a predicted range end counts as "open-ended". */
export const OPEN_END_YEAR = 2024

const platformKey = (value: string) => value.toLowerCase().replace(/^hbo\s+/, '').replace(/[^a-z0-9]/g, '')

/** Free-text viewing contexts (model mode) folded onto the app's two canonical values. */
export function normalizeViewingContext(value: string | undefined): string | undefined {
  if (!value) return undefined
  const lower = value.toLowerCase()
  if (lower === 'date-night' || /\bdate\b|\bcita\b/.test(lower)) return 'date-night'
  if (lower === 'movie-night' || /movie night|noche de cine/.test(lower)) return 'movie-night'
  return lower
}

export function fieldCorrect(field: Field, expected: ExpectedPreferences, predicted: ConversationPreferences): boolean {
  switch (field) {
    case 'genres':
    case 'excludedGenres':
    case 'moods':
      return sameSet(expected[field] ?? [], predicted[field] ?? [])
    case 'platforms':
      return sameSet((expected.platforms ?? []).map(platformKey), (predicted.platforms ?? []).map(platformKey))
    case 'languages':
      return sameSet((expected.languages ?? []).map((code) => code.toLowerCase()), (predicted.languages ?? []).map((code) => code.toLowerCase()))
    case 'maxRuntime':
    case 'peopleCount':
      return (expected[field] ?? undefined) === (predicted[field] ?? undefined)
    case 'viewingContext':
      return expected.viewingContext === normalizeViewingContext(predicted.viewingContext)
    case 'yearRange': {
      if (!expected.yearRange) return !predicted.yearRange
      if (!predicted.yearRange) return false
      const [start, end] = expected.yearRange
      const [predictedStart, predictedEnd] = predicted.yearRange
      if (predictedStart !== start) return false
      return end === null ? predictedEnd >= OPEN_END_YEAR : predictedEnd === end
    }
  }
}

export interface CaseScore {
  fields: Record<Field, boolean>
  exactMatch: boolean
  exclusionHits: number
  exclusionTotal: number
  /** A genre both requested and excluded, or a genre the user excluded showing up as requested. */
  contradiction: boolean
}

export function scoreCase(expected: ExpectedPreferences, predicted: ConversationPreferences): CaseScore {
  const fields = Object.fromEntries(FIELDS.map((field) => [field, fieldCorrect(field, expected, predicted)])) as Record<Field, boolean>
  const expectedExclusions = expected.excludedGenres ?? []
  const predictedExclusions = new Set(predicted.excludedGenres ?? [])
  const predictedGenres = predicted.genres ?? []
  const contradiction = predictedGenres.some((genre) => predictedExclusions.has(genre) || expectedExclusions.includes(genre))
  return {
    fields,
    exactMatch: FIELDS.every((field) => fields[field]),
    exclusionHits: expectedExclusions.filter((genre) => predictedExclusions.has(genre)).length,
    exclusionTotal: expectedExclusions.length,
    contradiction,
  }
}

export interface ExtractionSummary {
  cases: number
  fieldAccuracy: Record<Field, number>
  /** Micro-average over every (case, field) check. */
  overallFieldAccuracy: number
  exactMatch: number
  exclusionRecall: number
  contradictionRate: number
}

export function summarize(scores: CaseScore[]): ExtractionSummary {
  const share = (count: number, total: number) => total === 0 ? 0 : count / total
  const fieldAccuracy = Object.fromEntries(FIELDS.map((field) => [field, share(scores.filter((score) => score.fields[field]).length, scores.length)])) as Record<Field, number>
  const fieldChecks = scores.reduce((total, score) => total + FIELDS.filter((field) => score.fields[field]).length, 0)
  return {
    cases: scores.length,
    fieldAccuracy,
    overallFieldAccuracy: share(fieldChecks, scores.length * FIELDS.length),
    exactMatch: share(scores.filter((score) => score.exactMatch).length, scores.length),
    exclusionRecall: share(scores.reduce((total, score) => total + score.exclusionHits, 0), scores.reduce((total, score) => total + score.exclusionTotal, 0)),
    contradictionRate: share(scores.filter((score) => score.contradiction).length, scores.length),
  }
}
