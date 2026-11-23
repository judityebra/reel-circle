/**
 * Types and annotation guide for the hand-labelled extraction cases (English and Spanish, single-
 * and multi-turn). The cases themselves live in fixtures/extraction-cases.json, shared with the
 * Python recommendation agent; label new cases there following this guide.
 *
 * Annotation guide, applied before looking at any parser output:
 * - genres / excludedGenres: only genres the user explicitly asks for or rules out, using the
 *   app's canonical list. "funny" / "divertido" count as Comedy, "scary" as Horror,
 *   "animated" as Animation, "romantic" as Romance.
 * - moods: only when wording clearly names one of Gentle, Playful, Electric, Reflective, Twisty
 *   (cozy/calm -> Gentle, playful/lighthearted -> Playful, intense -> Electric,
 *   thoughtful -> Reflective, twisty/mind-bending -> Twisty). Vague words such as "darker" are
 *   left unlabelled.
 * - maxRuntime: minutes, only when the user states a cap.
 * - yearRange: [start, end]; `null` end means open-ended ("since 2010").
 * - languages: ISO 639-1 codes for the film's spoken language.
 * - peopleCount: total group size, only when stated unambiguously.
 * - viewingContext: 'date-night' or 'movie-night', only when stated.
 * - Multi-turn cases are labelled with the state the user would expect after the last turn.
 *   An explicit "start over" clears earlier turns; "actually X is fine" lifts an exclusion.
 * Any field not listed is expected to be empty.
 */

import cases from '../fixtures/extraction-cases.json' with { type: 'json' }

export interface ExpectedPreferences {
  genres?: string[]
  excludedGenres?: string[]
  moods?: string[]
  platforms?: string[]
  maxRuntime?: number
  yearRange?: [number, number | null]
  languages?: string[]
  peopleCount?: number
  viewingContext?: 'date-night' | 'movie-night'
}

export interface ExtractionCase {
  id: string
  lang: 'en' | 'es'
  turns: string[]
  expected: ExpectedPreferences
}

/** The cases live in fixtures/extraction-cases.json so the Python recommendation agent can share them. */
export const extractionCases: ExtractionCase[] = cases as ExtractionCase[]
