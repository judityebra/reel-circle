import type { ConversationPreferences } from './constraints'
import type { Movie, Profile } from './recommendation'

export interface AgentApiStep {
  tool: 'extract_preferences' | 'filter_candidates' | 'rank_candidates' | 'validate_results'
  status: 'completed' | 'skipped'
  durationMs: number
  observation: string
}

export interface AgentApiResponse {
  preferences: ConversationPreferences
  recommendations: Array<{
    movie: Movie
    score: number
    matchedPreferences: string[]
    explanation: string
  }>
  clarificationQuestion?: string
  meta: {
    extractionMode: 'model' | 'deterministic'
    modelStatus: 'not-configured' | 'failed' | 'succeeded'
    modelError?: string
    durationMs: number
    trace: AgentApiStep[]
  }
}

type Fetcher = (input: string, init: RequestInit) => Promise<Pick<Response, 'ok' | 'status' | 'json'>>

export async function requestAgentTurn(
  baseUrl: string,
  message: string,
  preferences: ConversationPreferences,
  candidates: Movie[],
  profiles: Profile[],
  fetcher: Fetcher = fetch,
): Promise<AgentApiResponse> {
  const response = await fetcher(`${baseUrl.replace(/\/$/, '')}/agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
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
