import { describe, expect, it } from 'vitest'
import type { RelaxationPause } from './agent-api'
import { describeNothingFits, describeRelaxation } from './relaxation-text'

describe('relaxation proposal text', () => {
  it('asks to drop the year range in English', () => {
    expect(describeRelaxation({ type: 'relaxation', constraint: 'yearRange', from: [1990, 1999], to: null }, 'en'))
      .toBe('Nothing fits every hard constraint. Allow any year instead of 1990–1999?')
  })

  it.each<[RelaxationPause, string]>([
    [{ type: 'relaxation', constraint: 'languages', from: ['ko'], to: [] }, 'Allow any language instead of KO?'],
    [{ type: 'relaxation', constraint: 'genres', from: ['Thriller', 'Crime'], to: [] }, 'Allow any genre instead of Thriller, Crime?'],
    [{ type: 'relaxation', constraint: 'maxRuntime', from: 90, to: 105 }, 'Allow up to 105 min instead of 90?'],
    [{ type: 'relaxation', constraint: 'platforms', from: ['Filmin'], to: [] }, 'Allow any platform instead of Filmin?'],
  ])('names the one constraint it proposes to loosen: %o', (pause, question) => {
    expect(describeRelaxation(pause, 'en')).toBe(`Nothing fits every hard constraint. ${question}`)
  })

  it('speaks Spanish with genres in the reader locale', () => {
    const spanish = (value: string) => ({ Thriller: 'Suspense' }[value] ?? value)
    expect(describeRelaxation({ type: 'relaxation', constraint: 'genres', from: ['Thriller'], to: [] }, 'es', spanish))
      .toBe('Nada cumple todas las condiciones. ¿Aceptas cualquier género en lugar de Suspense?')
  })
})

describe('nothing fits after every relaxation was rejected', () => {
  it.each([
    ['en', 'yearRange', 'Nothing fits, even after offering to loosen each constraint. The tightest one is the year range.'],
    ['es', 'excludedGenres', 'Nada encaja, ni ofreciendo relajar cada condición. La más restrictiva: los géneros excluidos.'],
    ['en', null, 'Nothing fits, even after offering to loosen each constraint.'],
  ] as const)('in %s names the tightest constraint (%s)', (locale, tightestConstraint, message) => {
    expect(describeNothingFits({ tightestConstraint }, locale)).toBe(message)
  })
})
