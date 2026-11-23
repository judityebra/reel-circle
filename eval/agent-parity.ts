/**
 * Parity fixtures for the recommendation agent.
 *
 * The fixtures freeze what the TypeScript pipeline (`/api/agent`, deterministic mode, local
 * semantic vectors) returns for a fixed candidate pool and group, turn by turn. They are a golden
 * master: the reference a port of the recommendation agent must reproduce, not a statement that
 * every answer is right. Extraction quality is measured separately by `npm run eval:extraction`.
 *
 *   npm run fixtures:parity            # regenerate fixtures/agent-parity.json
 */
import handler from '../api/agent'
import type { ConversationPreferences } from '../src/lib/constraints'
import { catalog, demoProfiles } from '../src/data'
import type { Movie, Profile } from '../src/lib/recommendation'
import { extractionCases } from './extraction-cases'
import { createRng } from './rng'
import { generateWorld } from './synthetic'

export type FixturePreferences = Omit<ConversationPreferences, 'lastUpdatedAt'>

export interface ExpectedRecommendation {
  movieId: string
  score: number
  matchedPreferences: string[]
  explanation: string
  semanticScore: number
  semanticTerms: string[]
}

export interface ExpectedTurn {
  preferences: FixturePreferences
  /** Movies in the candidate pool that meet every hard constraint. */
  survivors: number
  recommendations: ExpectedRecommendation[]
  violations: number
}

export interface ParityTurn {
  message: string
  expected: ExpectedTurn
}

export interface ParityConversation {
  id: string
  lang: 'en' | 'es'
  /** True when the conversation exercises semantic retrieval with vibe phrases. */
  vibe: boolean
  turns: ParityTurn[]
}

export interface ParityFixtures {
  description: string
  limit: number
  candidatePool: Movie[]
  persons: Profile[]
  conversations: ParityConversation[]
}

const FIXTURE_SEED = 2026

/** A shallow copy without one field, for dropping data the fixtures must not carry. */
function withoutKey<T>(value: object, key: string): T {
  const copy: Record<string, unknown> = { ...value }
  delete copy[key]
  return copy as T
}
const LIMIT = 5

const vibeConversations: Array<Omit<ParityConversation, 'turns'> & { messages: string[] }> = [
  { id: 'vibe-en-cozy-rainy', lang: 'en', vibe: true, messages: ['Something cozy and heartwarming for a rainy night'] },
  { id: 'vibe-en-dark-twist', lang: 'en', vibe: true, messages: ['A dark, mind-bending story with a twist ending'] },
  { id: 'vibe-en-space-friends', lang: 'en', vibe: true, messages: ['A fun space adventure to watch with friends'] },
  { id: 'vibe-es-tierno-familia', lang: 'es', vibe: true, messages: ['Algo tierno y nostálgico sobre la familia'] },
  { id: 'vibe-es-venganza', lang: 'es', vibe: true, messages: ['Una historia oscura de venganza'] },
  { id: 'vibe-en-mt-no-horror-twist', lang: 'en', vibe: true, messages: ['No horror tonight', 'Something with a clever twist, under two hours'] },
]

/** Removes every variable that would switch the pipeline away from deterministic, local mode. */
export function resetToDeterministicEnvironment() {
  for (const name of ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'EMBEDDING_PROVIDER', 'EMBEDDING_BASE_URL', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']) {
    delete process.env[name]
  }
}

/** Short synopses and keywords, so semantic retrieval has text to match (the curated catalog ships without them). */
const curatedText: Record<string, Pick<Movie, 'overview' | 'keywords'>> = {
  'aftersun-2022': { overview: 'A woman looks back on a holiday in Turkey she took at eleven with her young father, a tender and melancholy summer.', keywords: ['father', 'daughter', 'holiday', 'memory', 'nostalgia'] },
  'clue-1985': { overview: 'Six guests at a dinner party in a New England mansion become suspects when their blackmailer is murdered.', keywords: ['murder', 'mansion', 'dinner party', 'whodunit'] },
  'decision-2022': { overview: 'A detective investigating a death on a mountain becomes obsessed with the dead man\'s mysterious widow.', keywords: ['detective', 'widow', 'obsession', 'investigation'] },
  'perfect-days-2023': { overview: 'A Tokyo toilet cleaner finds quiet joy in his daily routine, his music, his books and the trees he photographs.', keywords: ['routine', 'solitude', 'tokyo', 'music'] },
  'rye-lane-2023': { overview: 'Two young Londoners reeling from bad breakups spend one day wandering South London together.', keywords: ['breakup', 'london', 'romance', 'friendship'] },
  'coherence-2013': { overview: 'On the night a comet passes, friends at a dinner party face strange events that split reality into parallel worlds.', keywords: ['comet', 'parallel realities', 'dinner party', 'twist'] },
  'past-lives-2023': { overview: 'Two childhood friends separated when one emigrates from Seoul meet again in New York decades later.', keywords: ['childhood friends', 'immigration', 'destiny', 'nostalgia'] },
  'attack-block-2011': { overview: 'A South London teenage gang defends their tower block from an alien invasion on Bonfire Night.', keywords: ['aliens', 'invasion', 'gang', 'london', 'friends'] },
}

/** The candidate pool: the curated catalog plus a seeded synthetic slate, without posters or latent quality. */
export function buildCandidatePool(): Movie[] {
  const world = generateWorld(createRng(FIXTURE_SEED), { movies: 60, users: 0, directors: 12 })
  const synthetic = world.movies.map((movie) => withoutKey<Movie>(movie, 'quality'))
  const curated = catalog.map((movie) => ({ ...withoutKey<Movie>(movie, 'poster'), ...curatedText[movie.id] }))
  return [...curated, ...synthetic]
}

/** Persons as the browser sends them: no ratings, no watch history, only learned weights. */
export function buildPersons(): Profile[] {
  const strip = ({ id, name, color, preferredGenres, tasteWeights, featureWeights }: Profile): Profile => ({
    id, name, color, preferredGenres, tasteWeights, featureWeights, ratings: [], watched: [],
  })
  return [
    ...demoProfiles.map(strip),
    strip({
      id: 'sam', name: 'Sam', color: '#4169c1', ratings: [], watched: [], preferredGenres: ['Thriller', 'Crime'],
      tasteWeights: { Horror: -1.2, Comedy: 0.6, Thriller: 0.9 },
      featureWeights: { 'language:Korean': 0.8, 'mood:Twisty': 0.5, 'decade:2010s': 0.3 },
    }),
  ]
}

interface CapturedResponse { status: number; body: unknown }

async function callAgent(body: Record<string, unknown>): Promise<CapturedResponse> {
  const captured: CapturedResponse = { status: 0, body: undefined }
  const response = {
    status(code: number) { captured.status = code; return response },
    json(payload: unknown) { captured.body = payload },
    setHeader() {},
  }
  await handler({ method: 'POST', body }, response)
  return captured
}

const round = (value: number) => Math.round(value * 10_000) / 10_000
const countFrom = (observation: string | undefined, pattern: RegExp) => Number(observation?.match(pattern)?.[1] ?? 0)

interface AgentResponseBody {
  preferences: ConversationPreferences
  recommendations: Array<{ movie: Movie; score: number; matchedPreferences: string[]; explanation: string; semanticScore: number; semanticTerms: string[] }>
  meta: { trace: Array<{ tool: string; observation: string }> }
}

/** Replays a conversation through `/api/agent` the way the browser does, carrying preferences between turns. */
export async function replayConversation(
  conversation: Pick<ParityConversation, 'turns'>,
  fixtures: Pick<ParityFixtures, 'candidatePool' | 'persons' | 'limit'>,
): Promise<ExpectedTurn[]> {
  const results: ExpectedTurn[] = []
  let preferences: ConversationPreferences | undefined
  for (const { message } of conversation.turns) {
    const { status, body } = await callAgent({ message, preferences, candidates: fixtures.candidatePool, profiles: fixtures.persons, limit: fixtures.limit })
    if (status !== 200) throw new Error(`Agent returned ${status} for "${message}"`)
    const payload = body as AgentResponseBody
    preferences = payload.preferences
    const stablePreferences = withoutKey<FixturePreferences>(payload.preferences, 'lastUpdatedAt')
    const trace = new Map(payload.meta.trace.map(({ tool, observation }) => [tool, observation]))
    results.push(JSON.parse(JSON.stringify({
      preferences: stablePreferences,
      survivors: countFrom(trace.get('filter_candidates'), /^(\d+)\//),
      recommendations: payload.recommendations.map(({ movie, score, matchedPreferences, explanation, semanticScore, semanticTerms }) => ({
        movieId: movie.id,
        score: round(score),
        matchedPreferences,
        explanation,
        semanticScore: round(semanticScore),
        semanticTerms,
      })),
      violations: countFrom(trace.get('validate_results'), /^(\d+) hard-constraint/),
    })) as ExpectedTurn)
  }
  return results
}

/** Builds the full fixture set by running every conversation through the current pipeline. */
export async function buildParityFixtures(): Promise<ParityFixtures> {
  resetToDeterministicEnvironment()
  const base = { limit: LIMIT, candidatePool: buildCandidatePool(), persons: buildPersons() }
  const sources = [
    ...extractionCases.map(({ id, lang, turns }) => ({ id, lang, vibe: false, messages: turns })),
    ...vibeConversations,
  ]
  const conversations: ParityConversation[] = []
  for (const { id, lang, vibe, messages } of sources) {
    const placeholder = { turns: messages.map((message) => ({ message, expected: undefined as unknown as ExpectedTurn })) }
    const expected = await replayConversation(placeholder, base)
    conversations.push({ id, lang, vibe, turns: messages.map((message, index) => ({ message, expected: expected[index] })) })
  }
  return {
    description: 'Golden master of the TypeScript recommendation agent (deterministic extraction, local semantic vectors). Regenerate with `npm run fixtures:parity`; a port must reproduce every turn.',
    ...base,
    conversations,
  }
}
