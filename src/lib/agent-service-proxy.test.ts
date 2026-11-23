import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import handler from '../../api/agent'
import type { Movie, Profile } from './recommendation'

const movies: Movie[] = [
  { id: 'clue', title: 'Clue', year: 1985, runtime: 94, genres: ['Comedy', 'Mystery'], platforms: ['Filmin'], rating: 3.8 },
  { id: 'aftersun', title: 'Aftersun', year: 2022, runtime: 102, genres: ['Drama'], platforms: ['Filmin'], rating: 4.2 },
]
const persons: Profile[] = [{ id: 'maya', name: 'Maya', color: '#f36b35', ratings: [], watched: [], preferredGenres: ['Comedy'] }]
const turn = { message: 'Something funny', candidates: movies, profiles: persons, limit: 3 }

const pythonPayload = {
  preferences: { genres: ['Comedy'], excludedGenres: [], platforms: [], moods: [], lastUpdatedAt: 1 },
  recommendations: [{ movie: movies[0], score: 77, matchedPreferences: [], explanation: 'From Python.', semanticScore: 0, semanticTerms: [] }],
  meta: { extractionMode: 'deterministic', modelStatus: 'not-configured', semantic: { retrievalMode: 'local', embeddingMode: 'local', embeddingModel: 'local-hash-v1', dimensions: 384 }, durationMs: 3, trace: [] },
}

interface AgentBody {
  recommendations: Array<{ movie: Movie; explanation: string }>
  meta: { agentService?: { mode: string; fallbackReason?: string } }
}

interface Captured { status: number; body: AgentBody; headers: Record<string, string> }

async function callRecommendationAgent(body: unknown): Promise<Captured> {
  const captured: Captured = { status: 0, body: { recommendations: [], meta: {} }, headers: {} }
  const response = {
    status(code: number) { captured.status = code; return response },
    json(payload: unknown) { captured.body = payload as AgentBody },
    setHeader(name: string, value: string) { captured.headers[name] = value },
  }
  await handler({ method: 'POST', body }, response)
  return captured
}

const jsonResponse = (status: number, body: unknown) => ({ ok: status >= 200 && status < 300, status, json: async () => body })

beforeEach(() => {
  for (const name of ['OPENAI_API_KEY', 'EMBEDDING_MODEL', 'EMBEDDING_BASE_URL', 'QDRANT_URL', 'AGENT_SERVICE_URL', 'AGENT_SERVICE_TIMEOUT_MS']) delete process.env[name]
})

afterEach(() => {
  vi.unstubAllGlobals()
  delete process.env.AGENT_SERVICE_URL
  delete process.env.AGENT_SERVICE_TIMEOUT_MS
})

describe('recommendation agent proxy to the Python service', () => {
  it('answers with the TypeScript pipeline and calls nothing when no service URL is set', async () => {
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    const { status, body } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(fetcher).not.toHaveBeenCalled()
    expect(body.recommendations[0].movie.id).toBe('clue')
    expect(body.meta.agentService).toBeUndefined()
  })

  it('forwards the bounded turn to the Python service and returns its answer', async () => {
    process.env.AGENT_SERVICE_URL = 'https://agent.example.test/'
    const fetcher = vi.fn(async () => jsonResponse(200, pythonPayload))
    vi.stubGlobal('fetch', fetcher)
    const { status, body, headers } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(headers['Cache-Control']).toBe('no-store')
    expect(fetcher).toHaveBeenCalledOnce()
    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://agent.example.test/agent')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual({ message: 'Something funny', candidates: movies, profiles: persons, limit: 3 })
    expect(body.recommendations[0].explanation).toBe('From Python.')
    expect(body.meta.agentService).toEqual({ mode: 'python' })
  })

  it.each([
    ['an HTTP error', async () => jsonResponse(503, { error: 'down' }), 'http-503'],
    ['an invalid answer', async () => jsonResponse(200, { recommendations: 'nope' }), 'invalid-response'],
    ['a network failure', async () => { throw new TypeError('fetch failed') }, 'unreachable'],
  ])('falls back to the TypeScript pipeline on %s and says so', async (_case, fetcher, reason) => {
    process.env.AGENT_SERVICE_URL = 'https://agent.example.test'
    vi.stubGlobal('fetch', vi.fn(fetcher))
    const { status, body } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(body.recommendations[0].movie.id).toBe('clue')
    expect(body.recommendations[0].explanation).not.toBe('From Python.')
    expect(body.meta.agentService).toEqual({ mode: 'typescript-fallback', fallbackReason: reason })
  })

  it('falls back when the Python service does not answer in time', async () => {
    process.env.AGENT_SERVICE_URL = 'https://agent.example.test'
    process.env.AGENT_SERVICE_TIMEOUT_MS = '20'
    vi.stubGlobal('fetch', vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'TimeoutError')))
    })))
    const { status, body } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(body.meta.agentService).toEqual({ mode: 'typescript-fallback', fallbackReason: 'timeout' })
  })

  it('still rejects a bad request before calling the Python service', async () => {
    process.env.AGENT_SERVICE_URL = 'https://agent.example.test'
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    const { status } = await callRecommendationAgent({ message: '' })
    expect(status).toBe(400)
    expect(fetcher).not.toHaveBeenCalled()
  })
})
