import type { ConversationPreferences } from './constraints'
import type { Movie, Profile } from './recommendation'

export interface AgentApiStep {
  tool: 'extract_preferences' | 'filter_candidates' | 'semantic_retrieval' | 'rank_candidates' | 'validate_results'
  status: 'completed' | 'skipped'
  durationMs: number
  observation: string
}

/** Why /api/agent answered with the TypeScript pipeline instead of the Python service. */
export type AgentServiceFallbackReason = 'timeout' | 'unreachable' | 'invalid-response' | `http-${number}`

export type AgentServiceInfo =
  | { mode: 'python' }
  | { mode: 'typescript-fallback'; fallbackReason: AgentServiceFallbackReason }

export interface AgentApiResponse {
  /** Set by the Python recommendation agent: send it back to continue the same remembered conversation. */
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

interface AgentTurnOptions {
  conversationId?: string
  fetcher?: Fetcher
}

export async function requestAgentTurn(
  baseUrl: string,
  message: string,
  preferences: ConversationPreferences,
  candidates: Movie[],
  profiles: Profile[],
  { conversationId, fetcher = fetch }: AgentTurnOptions = {},
): Promise<AgentApiResponse> {
  const response = await fetcher(`${baseUrl.replace(/\/$/, '')}/agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversationId,
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

/** Asks the server to forget a remembered conversation (best effort: used by Delete data and reset). */
export async function deleteAgentConversation(baseUrl: string, conversationId: string, fetcher: Fetcher = fetch): Promise<void> {
  await fetcher(`${baseUrl.replace(/\/$/, '')}/agent`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversationId }),
  })
}
