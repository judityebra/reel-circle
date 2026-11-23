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
import { scoreCandidates, type Movie, type Profile, type RecommendationOptions } from '../src/lib/recommendation'
import { extractionCases } from './extraction-cases'
import { createRng } from './rng'
import { generateWorld } from './synthetic'

export const PARITY_FIXTURES_URL = new URL('../fixtures/agent-parity.json', import.meta.url)

/** Conversation preferences without the wall-clock timestamp, which differs on every run. */
export type FixturePreferences = Omit<ConversationPreferences, 'lastUpdatedAt'>

export const HARD_CONSTRAINTS = ['maxRuntime', 'excludedGenres', 'yearRange', 'languages', 'genres', 'platforms'] as const
export type HardConstraint = typeof HARD_CONSTRAINTS[number]

/** What the browser sends for one turn, apart from the shared candidate pool, persons and limit. */
export interface TurnRequest {
  message: string
  /** Preferences carried from the previous turn; null on the first turn of a conversation. */
  preferences: FixturePreferences | null
}

export interface ExpectedRecommendation {
  movieId: string
  score: number
  matchedPreferences: string[]
  explanation: string
  semanticScore: number
  semanticTerms: string[]
}

export interface ConstraintViolation {
  movieId: string
  constraint: HardConstraint
}

export interface ExpectedTurn {
  preferences: FixturePreferences
  /** How many movies in the candidate pool meet every hard constraint. */
  candidatesMeetingConstraints: number
  recommendations: ExpectedRecommendation[]
  /** Recommended movies that break a hard constraint, checked one constraint at a time. */
  violations: ConstraintViolation[]
}

export interface ParityTurn {
  request: TurnRequest
  expected: ExpectedTurn
}

export interface ParityConversation {
  id: string
  lang: 'en' | 'es'
  /** True when the conversation uses vibe phrases to exercise semantic retrieval. */
  vibe: boolean
  turns: ParityTurn[]
}

/** The inputs shared by every turn: the candidate pool, the persons and the shortlist size. */
export interface ReplayContext {
  limit: number
  candidatePool: Movie[]
  persons: Profile[]
}

export interface ParityFixtures extends ReplayContext {
  description: string
  comparison: { scoreDecimals: number; explanations: 'exact' }
  conversations: ParityConversation[]
}

const FIXTURE_SEED = 2026
const LIMIT = 5
const SCORE_DECIMALS = 4

const vibeConversations = [
  { id: 'vibe-en-cozy-rainy', lang: 'en', messages: ['Something cozy and heartwarming for a rainy night'] },
  { id: 'vibe-en-dark-twist', lang: 'en', messages: ['A dark, mind-bending story with a twist ending'] },
  { id: 'vibe-en-space-friends', lang: 'en', messages: ['A fun space adventure to watch with friends'] },
  { id: 'vibe-es-tierno-familia', lang: 'es', messages: ['Algo tierno y nostálgico sobre la familia'] },
  { id: 'vibe-es-venganza', lang: 'es', messages: ['Una historia oscura de venganza'] },
  { id: 'vibe-en-mt-no-horror-twist', lang: 'en', messages: ['No horror tonight', 'Something with a clever twist, under two hours'] },
] as const

/**
 * Removes every variable that switches `/api/agent` away from deterministic, local mode: model
 * extraction (api/agent.ts), embedding providers (api/embedding-client.ts) and Qdrant
 * (api/qdrant-client.ts). Add any new provider variable here too.
 */
export function resetToDeterministicEnvironment() {
  for (const name of ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'EMBEDDING_PROVIDER', 'EMBEDDING_BASE_URL', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']) {
    delete process.env[name]
  }
}

function withoutKey<T extends object, K extends keyof T>(value: T, key: K): Omit<T, K> {
  const copy = { ...value }
  delete copy[key]
  return copy
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
function buildCandidatePool(): Movie[] {
  const world = generateWorld(createRng(FIXTURE_SEED), { movies: 60, users: 0, directors: 12 })
  const synthetic = world.movies.map((movie) => withoutKey(movie, 'quality'))
  const curated = catalog.map((movie) => ({ ...withoutKey(movie, 'poster'), ...curatedText[movie.id] }))
  return [...curated, ...synthetic]
}

/** Learned weights for the persons, so ranking exercises taste and feature weights for more than one person. */
const learnedWeights: Record<string, Pick<Profile, 'tasteWeights' | 'featureWeights'>> = {
  leo: { tasteWeights: { Comedy: 0.8, Mystery: 0.4, Drama: -0.5 }, featureWeights: { 'mood:Playful': 0.6, 'decade:1980s': 0.4 } },
  sam: { tasteWeights: { Horror: -1.2, Comedy: 0.6, Thriller: 0.9 }, featureWeights: { 'language:Korean': 0.8, 'mood:Twisty': 0.5, 'decade:2010s': 0.3 } },
}

/** Persons as the browser sends them: no ratings, no watch history, only preferred genres and learned weights. */
function buildPersons(): Profile[] {
  const sam = { id: 'sam', name: 'Sam', color: '#4169c1', preferredGenres: ['Thriller', 'Crime'] }
  return [...demoProfiles, sam].map(({ id, name, color, preferredGenres }) => ({
    id, name, color, preferredGenres, ...learnedWeights[id], ratings: [], watched: [],
  }))
}

/** The hard constraints a movie breaks, each checked on its own with the pipeline's own filter semantics. */
function brokenConstraints(movie: Movie, preferences: FixturePreferences): HardConstraint[] {
  return HARD_CONSTRAINTS.filter((constraint) => {
    const options: RecommendationOptions = { [constraint]: preferences[constraint] }
    return scoreCandidates([movie], [], options).length === 0
  })
}

interface RecommendationAgentResponse {
  status: number
  body?: {
    preferences: ConversationPreferences
    recommendations: Array<Omit<ExpectedRecommendation, 'movieId'> & { movie: Movie }>
  }
}

async function callRecommendationAgent(body: Record<string, unknown>): Promise<RecommendationAgentResponse> {
  const captured: RecommendationAgentResponse = { status: 0 }
  const response = {
    status(code: number) { captured.status = code; return response },
    json(payload: unknown) { captured.body = payload as RecommendationAgentResponse['body'] },
    setHeader() {},
  }
  await handler({ method: 'POST', body }, response)
  return captured
}

const round = (value: number) => Number(value.toFixed(SCORE_DECIMALS))

/**
 * Replays a conversation through `/api/agent` the way the browser does: each turn sends the
 * preferences the previous turn returned, with the shared candidate pool, persons and limit.
 */
export async function replayConversation(messages: readonly string[], context: ReplayContext): Promise<ParityTurn[]> {
  const turns: ParityTurn[] = []
  let carried: ConversationPreferences | undefined
  for (const message of messages) {
    const { status, body } = await callRecommendationAgent({ message, preferences: carried, candidates: context.candidatePool, profiles: context.persons, limit: context.limit })
    if (status !== 200 || !body) throw new Error(`Recommendation agent returned ${status} for "${message}"`)
    const preferences = withoutKey(body.preferences, 'lastUpdatedAt')
    const recommendations = body.recommendations.map(({ movie, score, matchedPreferences, explanation, semanticScore, semanticTerms }) => ({
      movieId: movie.id, score: round(score), matchedPreferences, explanation, semanticScore: round(semanticScore), semanticTerms,
    }))
    const violations = body.recommendations.flatMap(({ movie }) =>
      brokenConstraints(movie, preferences).map((constraint) => ({ movieId: movie.id, constraint })))
    const turn: ParityTurn = {
      request: { message, preferences: carried ? withoutKey(carried, 'lastUpdatedAt') : null },
      expected: {
        preferences,
        candidatesMeetingConstraints: scoreCandidates(context.candidatePool, context.persons, preferences).length,
        recommendations,
        violations,
      },
    }
    // A JSON round trip drops undefined fields, so a replayed turn compares equal to one read from disk.
    turns.push(JSON.parse(JSON.stringify(turn)) as ParityTurn)
    carried = body.preferences
  }
  return turns
}

/** Builds the full fixture set by running every conversation through the current pipeline. */
export async function buildParityFixtures(): Promise<ParityFixtures> {
  resetToDeterministicEnvironment()
  const context: ReplayContext = { limit: LIMIT, candidatePool: buildCandidatePool(), persons: buildPersons() }
  const scripted = [
    ...extractionCases.map(({ id, lang, turns }) => ({ id, lang, vibe: false, messages: turns })),
    ...vibeConversations.map(({ id, lang, messages }) => ({ id, lang, vibe: true, messages })),
  ]
  const conversations: ParityConversation[] = []
  for (const { id, lang, vibe, messages } of scripted) {
    conversations.push({ id, lang, vibe, turns: await replayConversation(messages, context) })
  }
  return {
    description: [
      'Golden master of the TypeScript recommendation agent (deterministic extraction, local semantic vectors, no model, no Qdrant).',
      'Replay each turn by POSTing { message, preferences, candidates: candidatePool, profiles: persons, limit }; preferences is request.preferences (null means omit), any lastUpdatedAt may be added.',
      'Compare expected.preferences ignoring lastUpdatedAt, recommendations in order with scores rounded to comparison.scoreDecimals, and explanations word for word.',
      'Regenerate with `npm run fixtures:parity` only when a behaviour change is intended.',
    ].join(' '),
    comparison: { scoreDecimals: SCORE_DECIMALS, explanations: 'exact' },
    ...context,
    conversations,
  }
}
