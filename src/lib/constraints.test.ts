import { describe, expect, it } from 'vitest'
import { createConversationPreferences, describeConstraints, parseConstraintPrompt, updateConversationPreferences } from './constraints'

describe('parseConstraintPrompt', () => {
  it('extracts Spanish runtime, platform, genre, and fairness constraints', () => {
    expect(parseConstraintPrompt('Algo divertido en Filmin de menos de 90 minutos que nadie odie')).toEqual({
      maxRuntime: 90,
      genres: ['Comedy'],
      platforms: ['Filmin'],
      moods: [],
      rankingMode: 'least-misery',
    })
  })

  it('extracts English mood and provider constraints', () => {
    expect(parseConstraintPrompt('A gentle romantic movie on Movistar Plus')).toEqual({
      genres: ['Romance'],
      platforms: ['Movistar Plus+'],
      moods: ['Gentle'],
    })
  })

  it('returns no constraints for unrelated text', () => {
    expect(parseConstraintPrompt('surprise us')).toEqual({ genres: [], platforms: [], moods: [] })
  })

  it('understands informal mood and broader genre language', () => {
    expect(parseConstraintPrompt('give me a super cunty horror movie')).toEqual({
      genres: ['Horror'],
      platforms: [],
      moods: ['Playful'],
    })
  })

  it('understands hour-based runtime in Spanish', () => {
    expect(parseConstraintPrompt('algo de menos de dos horas')).toEqual({
      maxRuntime: 120,
      genres: [],
      platforms: [],
      moods: [],
    })
  })

  it('describes parsed constraints for UI feedback', () => {
    expect(describeConstraints(parseConstraintPrompt('funny on Netflix under 90 minutes'))).toEqual([
      'Under 90 min',
      'Comedy',
      'Netflix',
    ])
    expect(describeConstraints(parseConstraintPrompt('surprise me'))).toEqual(['Any movie'])
  })

  it('preserves negative preferences across conversational refinements', () => {
    const firstTurn = updateConversationPreferences(createConversationPreferences(1), 'I do not want horror', 2)
    const secondTurn = updateConversationPreferences(firstTurn, 'Something darker, under 90 minutes', 3)

    expect(secondTurn).toMatchObject({
      genres: [],
      excludedGenres: ['Horror'],
      moods: ['Twisty'],
      maxRuntime: 90,
      lastUpdatedAt: 3,
    })
  })

  it('extracts group context, language, and release period', () => {
    const preferences = updateConversationPreferences(createConversationPreferences(), 'Movie night with 4 friends, French films from the 1990s')

    expect(preferences).toMatchObject({
      peopleCount: 4,
      viewingContext: 'movie-night',
      languages: ['fr'],
      yearRange: [1990, 1999],
    })
  })
})