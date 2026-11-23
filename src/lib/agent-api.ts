import type { ConversationPreferences } from './constraints'
import type { Movie, Profile } from './recommendation'

export interface AgentApiStep {
  tool: 'extract_preferences' | 'filter_candidates' | 'semantic_retrieval' | 'rank_candidates' | 'validate_results' | 'propose_relaxation' | 'step_limit'
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

export type RelaxationDecision = 'accept' | 'reject'

export const isRelaxationDecision = (value: unknown): value is RelaxationDecision => value === 'accept' || value === 'reject'

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
  /** Present while the agent waits for the user to accept or reject a relaxation. */
  pause?: RelaxationPause
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
  /** The answer to a pending relaxation; without a pending pause the server treats the turn as ordinary. */
  decision?: RelaxationDecision
  fetcher?: Fetcher
}

export async function requestAgentTurn(
  baseUrl: string,
  message: string,
  preferences: ConversationPreferences,
  candidates: Movie[],
  profiles: Profile[],
  { conversationId, decision, fetcher = fetch }: AgentTurnOptions = {},
): Promise<AgentApiResponse> {
  const response = await fetcher(agentUrl(baseUrl), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversationId,
      decision,
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
  if (!payload?.preferences || !Array.isArray(payload.recommendations) || !Array.isArray(payload.meta?.trace)) {
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
