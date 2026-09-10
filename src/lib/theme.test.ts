import { describe, expect, it } from 'vitest'
import { resolveTheme, type ThemePreference } from './theme'

describe('resolveTheme', () => {
  it.each<[ThemePreference, boolean, 'light' | 'dark']>([
    ['light', true, 'light'],
    ['dark', false, 'dark'],
    ['system', true, 'dark'],
    ['system', false, 'light'],
  ])('resolves %s with dark preference %s to %s', (preference, prefersDark, expected) => {
    expect(resolveTheme(preference, prefersDark)).toBe(expected)
  })
})