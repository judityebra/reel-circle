import { describe, expect, it } from 'vitest'
import { describeConstraints, parseConstraintPrompt } from './constraints'

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
})