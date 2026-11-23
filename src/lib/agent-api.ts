import type { ConversationPreferences } from './constraints'
import type { Movie, Profile } from './recommendation'

export interface AgentApiStep {
  tool: 'extract_preferences' | 'filter_candidates' | 'semantic_retrieval' | 'rank_candidates' | 'validate_results' | 'ask_clarifying_question' | 'propose_relaxation' | 'step_limit'
  status: 'completed' | 'skipped'
  durationMs: number
  observation: string
}

/** The hard constraints the recommendation agent may offer to loosen, in the order it offers them. */
export type RelaxableConstraint = 'yearRange' | 'languages' | 'genres' | 'maxRuntime' | 'platforms'

/** The agent paused: nothing fits, and it asks to loosen one hard constraint. `to: null` means drop it. */
export type RelaxationPause = {
  [Constraint in RelaxableConstraint]: {
    type: 'relaxation'
    constraint: Constraint
    from: NonNullable<ConversationPreferences[Constraint]>
    to: ConversationPreferences[Constraint] | null
  }
}[RelaxableConstraint]

/** The agent paused: a model found nothing usable in the turn, so it asks. The reply is a new turn. */
export type ClarifyingPause = { type: 'clarifying_question'; question: string }

export type AgentPause = RelaxationPause | ClarifyingPause

export type RelaxationDecision = 'accept' | 'reject'

export const isRelaxationDecision = (value: unknown): value is RelaxationDecision => value === 'accept' || value === 'reject'

const relaxableConstraints: readonly string[] = ['yearRange', 'languages', 'genres', 'maxRuntime', 'platforms']

/** Whether a value is a pause the browser can render: a relaxation proposal or a clarifying question. */
const isAgentPause = (value: unknown): value is AgentPause => isRelaxationPause(value)
  || (!!value && typeof value === 'object' && (value as Record<string, unknown>).type === 'clarifying_question'
    && typeof (value as Record<string, unknown>).question === 'string')

/** Whether a value is a relaxation proposal the browser can render and answer. */
export const isRelaxationPause = (value: unknown): value is RelaxationPause => {
  if (!value || typeof value !== 'object') return false
  const pause = value as Record<string, unknown>
  return pause.type === 'relaxation' && typeof pause.constraint === 'string' && relaxableConstraints.includes(pause.constraint)
    && pause.from !== undefined && pause.from !== null && 'to' in pause
}

/** The preferences after an accepted relaxation: the constraint loosened as proposed (`to: null` drops it). */
export function applyRelaxation(preferences: ConversationPreferences, pause: RelaxationPause): ConversationPreferences {
  const relaxed: ConversationPreferences = { ...preferences }
  if (pause.to === null) delete relaxed[pause.constraint]
  else Object.assign(relaxed, { [pause.constraint]: pause.to })
  return relaxed
}

/** Why /api/agent answered with the TypeScript pipeline instead of the Python service. */
export type AgentServiceFallbackReason = 'timeout' | 'unreachable' | 'invalid-response' | `http-${number}`

export type AgentServiceInfo =
  | { mode: 'python' }
  | { mode: 'typescript-fallback'; fallbackReason: AgentServiceFallbackReason }

export interface AgentApiResponse {
  /** The server-side conversation: send it back to continue the same remembered conversation. */
  conversationId?: string
  preferences: ConversationPreferences
  recommendations: Array<{
    movie: Movie
    score: number
    matchedPreferences: string[]
    explanation: string
    semanticScore: number
    semanticTerms: string[]
  }>
  clarificationQuestion?: string
  /** Present while the agent waits: for an accept or reject of a relaxation, or for the answer to a question. */
  pause?: AgentPause
  /** Present when every relaxation was rejected: the constraint that let the fewest movies through. */
  nothingFits?: { tightestConstraint: RelaxableConstraint | 'excludedGenres' | null }
  meta: {
    extractionMode: 'model' | 'deterministic'
    modelStatus: 'not-configured' | 'failed' | 'succeeded'
    modelError?: string
    semantic: {
      retrievalMode: 'qdrant' | 'embedding-memory' | 'local'
      embeddingMode: 'provider' | 'local'
      embeddingModel: string
      dimensions: number
      fallbackReason?: 'embedding-failed' | 'qdrant-failed'
    }
    durationMs: number
    trace: AgentApiStep[]
    /** Added by the Python recommendation agent: movies in the candidate pool that meet every hard constraint. */
    candidatesMeetingConstraints?: number
    /** Present when AGENT_SERVICE_URL is configured: who answered the turn. */
    agentService?: AgentServiceInfo
  }
}

type Fetcher = (input: string, init: RequestInit) => Promise<Pick<Response, 'ok' | 'status' | 'json'>>

const agentUrl = (baseUrl: string) => `${baseUrl.replace(/\/$/, '')}/agent`

interface AgentTurnOptions {
  conversationId?: string
  /** The answer to a relaxation, with the proposal it answers; if that is not the pending pause, the turn is ordinary. */
  answer?: { decision: RelaxationDecision; answering: RelaxationPause }
  fetcher?: Fetcher
}

export async function requestAgentTurn(
  baseUrl: string,
  message: string,
  preferences: ConversationPreferences,
  candidates: Movie[],
  profiles: Profile[],
  { conversationId, answer, fetcher = fetch }: AgentTurnOptions = {},
): Promise<AgentApiResponse> {
  const response = await fetcher(agentUrl(baseUrl), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversationId,
      decision: answer?.decision,
      answering: answer?.answering,
      preferences,
      candidates: candidates.slice(0, 250),
      profiles: profiles.slice(0, 12).map(({ id, name, color, preferredGenres, tasteWeights, featureWeights }) => ({
        id,
        name,
        color,
        preferredGenres,
        tasteWeights,
        featureWeights,
        ratings: [],
        watched: [],
      })),
      limit: 5,
    }),
  })
  if (!response.ok) throw new Error(`Agent service unavailable (${response.status})`)
  const payload = await response.json() as AgentApiResponse
  if (!payload?.preferences || !Array.isArray(payload.recommendations) || !Array.isArray(payload.meta?.trace)
    || (payload.pause !== undefined && !isAgentPause(payload.pause))) {
    throw new Error('Agent service returned invalid data')
  }
  return payload
}

/** Asks the server to forget a remembered conversation; throws if it could not (Delete data reports that). */
export async function deleteAgentConversation(baseUrl: string, conversationId: string, fetcher: Fetcher = fetch): Promise<void> {
  const response = await fetcher(agentUrl(baseUrl), {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversationId }),
  })
  if (!response.ok) throw new Error(`Agent service could not forget the conversation (${response.status})`)
}
