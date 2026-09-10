import { describe, expect, it } from 'vitest'
import { resolveLocale, translate } from './i18n'

describe('i18n', () => {
  it('resolves supported browser and stored locales', () => {
    expect(resolveLocale('es', 'en-US')).toBe('es')
    expect(resolveLocale(null, 'es-ES')).toBe('es')
    expect(resolveLocale(null, 'fr-FR')).toBe('en')
  })

  it('translates known keys and falls back to English', () => {
    expect(translate('es', 'compareMovies')).toBe('Comparar dos películas')
    expect(translate('en', 'compareMovies')).toBe('Compare two movies')
  })
})