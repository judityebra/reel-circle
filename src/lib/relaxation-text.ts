import type { AgentApiResponse, RelaxableConstraint, RelaxationPause } from './agent-api'
import type { Locale } from './i18n'

type Format = (value: string) => string

interface Wording {
  noFit: string
  allowAny: (what: string, current: string) => string
  allowRuntime: (to: number, from: number) => string
  nothingFits: string
  tightest: (name: string) => string
  /** What "any ..." loosens to, for every constraint dropped outright. */
  anyOf: Record<Exclude<RelaxableConstraint, 'maxRuntime'>, string>
  /** How the "nothing fits" answer names a constraint. */
  names: Record<RelaxableConstraint | 'excludedGenres', string>
}

const wording: Record<Locale, Wording> = {
  en: {
    noFit: 'Nothing fits every hard constraint.',
    allowAny: (what, current) => `Allow any ${what} instead of ${current}?`,
    allowRuntime: (to, from) => `Allow up to ${to} min instead of ${from}?`,
    nothingFits: 'Nothing fits, even after offering to loosen each constraint.',
    tightest: (name) => `The tightest one is ${name}.`,
    anyOf: { yearRange: 'year', languages: 'language', genres: 'genre', platforms: 'platform' },
    names: {
      yearRange: 'the year range',
      languages: 'the languages',
      genres: 'the requested genres',
      maxRuntime: 'the maximum runtime',
      platforms: 'the platforms',
      excludedGenres: 'the excluded genres',
    },
  },
  es: {
    noFit: 'Nada cumple todas las condiciones.',
    allowAny: (what, current) => `¿Aceptas cualquier ${what} en lugar de ${current}?`,
    allowRuntime: (to, from) => `¿Aceptas hasta ${to} min en lugar de ${from}?`,
    nothingFits: 'Nada encaja, ni ofreciendo relajar cada condición.',
    tightest: (name) => `La más restrictiva: ${name}.`,
    anyOf: { yearRange: 'año', languages: 'idioma', genres: 'género', platforms: 'plataforma' },
    names: {
      yearRange: 'el rango de años',
      languages: 'los idiomas',
      genres: 'los géneros pedidos',
      maxRuntime: 'la duración máxima',
      platforms: 'las plataformas',
      excludedGenres: 'los géneros excluidos',
    },
  },
}

function question(pause: RelaxationPause, words: Wording, format: Format): string {
  switch (pause.constraint) {
    case 'maxRuntime':
      return words.allowRuntime(pause.to ?? pause.from, pause.from)
    case 'yearRange':
      return words.allowAny(words.anyOf.yearRange, pause.from.join('–'))
    case 'languages':
      return words.allowAny(words.anyOf.languages, pause.from.map((language) => language.toUpperCase()).join(', '))
    default:
      return words.allowAny(words.anyOf[pause.constraint], pause.from.map(format).join(', '))
  }
}

/** The agent's relaxation proposal as one sentence the user can accept or reject. */
export function describeRelaxation(pause: RelaxationPause, locale: Locale, format: Format = (value) => value): string {
  const words = wording[locale]
  return `${words.noFit} ${question(pause, words, format)}`
}

/** The honest answer once every relaxation was rejected, naming the constraint that blocks the most. */
export function describeNothingFits({ tightestConstraint }: NonNullable<AgentApiResponse['nothingFits']>, locale: Locale): string {
  const words = wording[locale]
  return tightestConstraint ? `${words.nothingFits} ${words.tightest(words.names[tightestConstraint])}` : words.nothingFits
}
