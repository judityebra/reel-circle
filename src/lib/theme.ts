export type ThemePreference = 'light' | 'dark' | 'system'

export function resolveTheme(
  preference: ThemePreference,
  prefersDark: boolean,
): 'light' | 'dark' {
  return preference === 'system' ? (prefersDark ? 'dark' : 'light') : preference
}