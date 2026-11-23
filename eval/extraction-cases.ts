/**
 * Hand-written extraction test prompts (English and Spanish, single- and multi-turn).
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

export const extractionCases: ExtractionCase[] = [
  // English, single turn
  { id: 'en-funny-90', lang: 'en', turns: ['Something funny under 90 minutes'], expected: { genres: ['Comedy'], maxRuntime: 90 } },
  { id: 'en-no-horror-mystery', lang: 'en', turns: ['No horror please, we want a mystery'], expected: { genres: ['Mystery'], excludedGenres: ['Horror'] } },
  { id: 'en-romance-date', lang: 'en', turns: ['A romantic movie for date night'], expected: { genres: ['Romance'], viewingContext: 'date-night' } },
  { id: 'en-scifi-netflix', lang: 'en', turns: ['Sci-fi on Netflix'], expected: { genres: ['Sci-Fi'], platforms: ['Netflix'] } },
  { id: 'en-90s-thriller', lang: 'en', turns: ['Anything from the 90s that is a thriller'], expected: { genres: ['Thriller'], yearRange: [1990, 1999] } },
  { id: 'en-cozy-animated', lang: 'en', turns: ['A cozy animated film for a rainy afternoon'], expected: { genres: ['Animation'], moods: ['Gentle'] } },
  { id: 'en-korean-crime-2h', lang: 'en', turns: ['Korean crime movie, nothing over two hours'], expected: { genres: ['Crime'], languages: ['ko'], maxRuntime: 120 } },
  { id: 'en-four-people-no-horror', lang: 'en', turns: ['We\'re 4 people and nobody likes horror'], expected: { peopleCount: 4, excludedGenres: ['Horror'] } },
  { id: 'en-mind-bending-since', lang: 'en', turns: ['Something mind-bending, released since 2010'], expected: { moods: ['Twisty'], yearRange: [2010, null] } },
  { id: 'en-doc-mubi', lang: 'en', turns: ['A documentary on MUBI'], expected: { genres: ['Documentary'], platforms: ['MUBI'] } },
  { id: 'en-not-drama-action', lang: 'en', turns: ['Not in the mood for drama. Something with action'], expected: { genres: ['Action'], excludedGenres: ['Drama'] } },
  { id: 'en-thoughtful-french', lang: 'en', turns: ['A thoughtful French film'], expected: { moods: ['Reflective'], languages: ['fr'] } },
  { id: 'en-three-intense-action', lang: 'en', turns: ['Me and two friends want an intense action movie'], expected: { peopleCount: 3, genres: ['Action'], moods: ['Electric'] } },
  { id: 'en-movie-night-scary-prime', lang: 'en', turns: ['Movie night! Something scary on Prime Video'], expected: { viewingContext: 'movie-night', genres: ['Horror'], platforms: ['Prime Video'] } },
  { id: 'en-fantasy-avoid-animated', lang: 'en', turns: ['I\'d like a fantasy film, but please avoid anything animated'], expected: { genres: ['Fantasy'], excludedGenres: ['Animation'] } },
  { id: 'en-japanese-animation-100', lang: 'en', turns: ['Japanese animation under 100 minutes'], expected: { genres: ['Animation'], languages: ['ja'], maxRuntime: 100 } },
  { id: 'en-comedy-or-romance-partner', lang: 'en', turns: ['A comedy or a romance, it\'s for date night with my partner'], expected: { genres: ['Comedy', 'Romance'], viewingContext: 'date-night' } },
  { id: 'en-playful-disney', lang: 'en', turns: ['Something playful and lighthearted on Disney+'], expected: { moods: ['Playful'], platforms: ['Disney+'] } },
  { id: 'en-twisty-mystery-2000s', lang: 'en', turns: ['Give me a twisty mystery from the 2000s'], expected: { genres: ['Mystery'], moods: ['Twisty'], yearRange: [2000, 2009] } },
  { id: 'en-without-scifi-horror', lang: 'en', turns: ['Without sci-fi or horror, please'], expected: { excludedGenres: ['Sci-Fi', 'Horror'] } },
  { id: 'en-spanish-thriller-95', lang: 'en', turns: ['A Spanish-language thriller that\'s at most 95 minutes'], expected: { genres: ['Thriller'], languages: ['es'], maxRuntime: 95 } },
  { id: 'en-hour-and-half', lang: 'en', turns: ['An hour and a half max, any genre'], expected: { maxRuntime: 90 } },
  { id: 'en-max-or-filmin', lang: 'en', turns: ['Something on HBO Max or Filmin'], expected: { platforms: ['Max', 'Filmin'] } },
  { id: 'en-five-crime', lang: 'en', turns: ['We are five people looking for a crime film'], expected: { peopleCount: 5, genres: ['Crime'] } },
  { id: 'en-anything-but-comedy', lang: 'en', turns: ['Anything but comedy'], expected: { excludedGenres: ['Comedy'] } },

  // Spanish, single turn
  { id: 'es-comedia-90', lang: 'es', turns: ['Una comedia de menos de 90 minutos'], expected: { genres: ['Comedy'], maxRuntime: 90 } },
  { id: 'es-nada-terror-misterio', lang: 'es', turns: ['Nada de terror, por favor. Algo de misterio'], expected: { genres: ['Mystery'], excludedGenres: ['Horror'] } },
  { id: 'es-romantica-cita', lang: 'es', turns: ['Una película romántica para una cita'], expected: { genres: ['Romance'], viewingContext: 'date-night' } },
  { id: 'es-cifi-netflix', lang: 'es', turns: ['Ciencia ficción en Netflix'], expected: { genres: ['Sci-Fi'], platforms: ['Netflix'] } },
  { id: 'es-thriller-90', lang: 'es', turns: ['Un thriller de los 90'], expected: { genres: ['Thriller'], yearRange: [1990, 1999] } },
  { id: 'es-tranquilo-animacion', lang: 'es', turns: ['Algo tranquilo y acogedor de animación'], expected: { genres: ['Animation'], moods: ['Gentle'] } },
  { id: 'es-cuatro-accion', lang: 'es', turns: ['Somos 4 personas y queremos algo de acción'], expected: { peopleCount: 4, genres: ['Action'] } },
  { id: 'es-coreana-crimen-2h', lang: 'es', turns: ['Una película coreana de crimen, máximo dos horas'], expected: { genres: ['Crime'], languages: ['ko'], maxRuntime: 120 } },
  { id: 'es-documental-filmin', lang: 'es', turns: ['Un documental en Filmin'], expected: { genres: ['Documentary'], platforms: ['Filmin'] } },
  { id: 'es-reflexivo-frances', lang: 'es', turns: ['Algo reflexivo en francés'], expected: { moods: ['Reflective'], languages: ['fr'] } },
  { id: 'es-noche-cine-terror-prime', lang: 'es', turns: ['Noche de cine con amigos: algo de terror en Prime Video'], expected: { viewingContext: 'movie-night', genres: ['Horror'], platforms: ['Prime Video'] } },
  { id: 'es-sin-drama-fantasia', lang: 'es', turns: ['Sin drama, quiero fantasía'], expected: { genres: ['Fantasy'], excludedGenres: ['Drama'] } },
  { id: 'es-espanola-2015', lang: 'es', turns: ['Una película en español posterior a 2015'], expected: { languages: ['es'], yearRange: [2015, null] } },
  { id: 'es-divertido-disney-90', lang: 'es', turns: ['Algo divertido en Disney+ que dure menos de hora y media'], expected: { genres: ['Comedy'], platforms: ['Disney+'], maxRuntime: 90 } },
  { id: 'es-ni-animacion-ni-terror', lang: 'es', turns: ['Que no sea de animación ni de terror'], expected: { excludedGenres: ['Animation', 'Horror'] } },
  { id: 'es-misterio-retorcido-movistar', lang: 'es', turns: ['Un misterio retorcido en Movistar'], expected: { genres: ['Mystery'], moods: ['Twisty'], platforms: ['Movistar Plus+'] } },
  { id: 'es-tres-comedia-romantica', lang: 'es', turns: ['Somos tres y queremos una comedia romántica'], expected: { peopleCount: 3, genres: ['Comedy', 'Romance'] } },

  // English, multi-turn
  { id: 'en-mt-funny-then-runtime', lang: 'en', turns: ['Something funny', 'Actually, make it under 100 minutes'], expected: { genres: ['Comedy'], maxRuntime: 100 } },
  { id: 'en-mt-no-horror-darker', lang: 'en', turns: ['No horror', 'Something darker, under 90 minutes'], expected: { excludedGenres: ['Horror'], maxRuntime: 90 } },
  { id: 'en-mt-start-over', lang: 'en', turns: ['A sci-fi movie on Netflix', 'Forget that, start over. A romance on MUBI'], expected: { genres: ['Romance'], platforms: ['MUBI'] } },
  { id: 'en-mt-lift-exclusion', lang: 'en', turns: ['No comedy tonight', 'Actually comedy is fine, something funny under 95 minutes'], expected: { genres: ['Comedy'], maxRuntime: 95 } },
  { id: 'en-mt-three-refinements', lang: 'en', turns: ['Japanese films please', 'Make it animated', 'And under two hours'], expected: { languages: ['ja'], genres: ['Animation'], maxRuntime: 120 } },
  { id: 'en-mt-group-thriller-year', lang: 'en', turns: ['We\'re 3 people', 'We want a thriller', 'Nothing older than 2000'], expected: { peopleCount: 3, genres: ['Thriller'], yearRange: [2000, null] } },

  // Spanish, multi-turn
  { id: 'es-mt-misterio-runtime', lang: 'es', turns: ['Algo de misterio', 'Mejor de menos de 100 minutos'], expected: { genres: ['Mystery'], maxRuntime: 100 } },
  { id: 'es-mt-exclusion-persists', lang: 'es', turns: ['Nada de terror', 'Ahora quiero algo de ciencia ficción en Filmin'], expected: { genres: ['Sci-Fi'], excludedGenres: ['Horror'], platforms: ['Filmin'] } },
  { id: 'es-mt-empezar-de-nuevo', lang: 'es', turns: ['Una comedia en Netflix', 'Empezar de nuevo: un drama coreano'], expected: { genres: ['Drama'], languages: ['ko'] } },
  { id: 'es-mt-lift-exclusion', lang: 'es', turns: ['Sin comedia', 'Bueno, la comedia sí vale, que sea romántica'], expected: { genres: ['Comedy', 'Romance'] } },
]
