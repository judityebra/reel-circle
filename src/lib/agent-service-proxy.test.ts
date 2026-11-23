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

async function callHandler(method: string, body: unknown): Promise<Captured> {
  const captured: Captured = { status: 0, body: { recommendations: [], meta: {} }, headers: {} }
  const response = {
    status(code: number) { captured.status = code; return response },
    json(payload: unknown) { captured.body = payload as AgentBody },
    setHeader(name: string, value: string) { captured.headers[name] = value },
  }
  await handler({ method, body }, response)
  return captured
}

const callRecommendationAgent = (body: unknown) => callHandler('POST', body)

const jsonResponse = (status: number, body: unknown) => ({ ok: status >= 200 && status < 300, status, json: async () => body })

type ServiceReply = (url: string, init: RequestInit) => Promise<unknown>

/** Stubs fetch as the Python service (answering like it by default); `url: null` means no service is configured. */
function fakeAgentService({ url = 'https://agent.example.test', reply = async () => jsonResponse(200, pythonPayload) }: { url?: string | null; reply?: ServiceReply } = {}) {
  if (url) process.env.AGENT_SERVICE_URL = url
  const fetcher = vi.fn(reply)
  vi.stubGlobal('fetch', fetcher)
  return fetcher
}

function firstRequest(fetcher: ReturnType<typeof fakeAgentService>) {
  const [url, init] = fetcher.mock.calls[0]
  return { url, init, body: JSON.parse(String(init.body ?? 'null')) }
}

const environmentNames = ['OPENAI_API_KEY', 'EMBEDDING_MODEL', 'EMBEDDING_BASE_URL', 'QDRANT_URL', 'AGENT_SERVICE_URL', 'AGENT_SERVICE_TIMEOUT_MS']
const originalEnvironment = Object.fromEntries(environmentNames.map((name) => [name, process.env[name]]))

beforeEach(() => {
  for (const name of environmentNames) delete process.env[name]
})

afterEach(() => {
  vi.unstubAllGlobals()
  for (const [name, value] of Object.entries(originalEnvironment)) {
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
})

describe('recommendation agent proxy to the Python service', () => {
  it('answers with the TypeScript pipeline and calls nothing when no service URL is set', async () => {
    const fetcher = fakeAgentService({ url: null })
    const { status, body } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(fetcher).not.toHaveBeenCalled()
    expect(body.recommendations[0].movie.id).toBe('clue')
    expect(body.meta.agentService).toBeUndefined()
  })

  it('forwards the bounded turn to the Python service and returns its answer', async () => {
    const fetcher = fakeAgentService({ url: 'https://agent.example.test/' })
    const { status, body, headers } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(headers['Cache-Control']).toBe('no-store')
    expect(fetcher).toHaveBeenCalledOnce()
    const { url, init } = firstRequest(fetcher)
    expect(url).toBe('https://agent.example.test/agent')
    expect(init.method).toBe('POST')
    expect(firstRequest(fetcher).body).toEqual({ message: 'Something funny', candidates: movies, profiles: persons, limit: 3 })
    expect(body.recommendations[0].explanation).toBe('From Python.')
    expect(body.meta.agentService).toEqual({ mode: 'python' })
  })

  it('never forwards ratings or watch history, whatever the caller sends', async () => {
    const fetcher = fakeAgentService()
    const leakyPerson = {
      ...persons[0],
      tasteWeights: { Comedy: 1 },
      featureWeights: { 'mood:Playful': 0.5 },
      ratings: [{ title: 'Moonlight', year: 2016, rating: 5 }],
      watched: [{ title: 'Moonlight', year: 2016 }],
      email: 'maya@example.test',
    }
    await callRecommendationAgent({ ...turn, profiles: [leakyPerson] })
    expect(firstRequest(fetcher).body.profiles).toEqual([{
      id: 'maya', name: 'Maya', color: '#f36b35', preferredGenres: ['Comedy'],
      tasteWeights: { Comedy: 1 }, featureWeights: { 'mood:Playful': 0.5 }, ratings: [], watched: [],
    }])
  })

  it('ignores an unusable timeout setting and uses the default', async () => {
    process.env.AGENT_SERVICE_TIMEOUT_MS = '-5'
    fakeAgentService()
    const { body } = await callRecommendationAgent(turn)
    expect(body.meta.agentService).toEqual({ mode: 'python' })
  })

  it.each([
    ['an HTTP error', async () => jsonResponse(503, { error: 'down' }), 'http-503'],
    ['an invalid answer', async () => jsonResponse(200, { recommendations: 'nope' }), 'invalid-response'],
    ['an answer that is not JSON', async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError('Unexpected token') } }), 'invalid-response'],
    ['a recommendation without an explanation', async () => jsonResponse(200, {
      ...pythonPayload,
      recommendations: [{ movie: movies[0], score: 77 }],
    }), 'invalid-response'],
    ['a network failure', async () => { throw new TypeError('fetch failed') }, 'unreachable'],
  ])('falls back to the TypeScript pipeline on %s and says so', async (_case, fetcher, reason) => {
    fakeAgentService({ reply: fetcher })
    const { status, body } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(body.recommendations[0].movie.id).toBe('clue')
    expect(body.recommendations[0].explanation).not.toBe('From Python.')
    expect(body.meta.agentService).toEqual({ mode: 'typescript-fallback', fallbackReason: reason })
  })

  it('falls back when the Python service does not answer in time', async () => {
    process.env.AGENT_SERVICE_TIMEOUT_MS = '20'
    fakeAgentService({ reply: (_url, init) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'TimeoutError')))
    }) })
    const { status, body } = await callRecommendationAgent(turn)
    expect(status).toBe(200)
    expect(body.meta.agentService).toEqual({ mode: 'typescript-fallback', fallbackReason: 'timeout' })
  })

  it('still rejects a bad request before calling the Python service', async () => {
    const fetcher = fakeAgentService()
    const { status } = await callRecommendationAgent({ message: '' })
    expect(status).toBe(400)
    expect(fetcher).not.toHaveBeenCalled()
  })
})

describe('conversations remembered by the Python service', () => {
  it('forwards the conversation id and keeps it when the TypeScript fallback answers', async () => {
    const fetcher = fakeAgentService({ reply: async () => jsonResponse(503, {}) })
    const { body } = await callRecommendationAgent({ ...turn, conversationId: 'conv-123' })
    expect(firstRequest(fetcher).body.conversationId).toBe('conv-123')
    expect((body as unknown as { conversationId?: string }).conversationId).toBe('conv-123')
  })

  it('rejects a malformed conversation id, like the Python service', async () => {
    const fetcher = fakeAgentService()
    const { status } = await callRecommendationAgent({ ...turn, conversationId: '../../etc/passwd' })
    expect(status).toBe(400)
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each([
    ['accept', 'accept'],
    ['reject', 'reject'],
    ['anything else', undefined],
  ])('forwards a relaxation decision of %s to the Python service only when it is valid', async (decision, forwarded) => {
    const fetcher = fakeAgentService()
    await callRecommendationAgent({ ...turn, conversationId: 'conv-123', decision })
    expect(firstRequest(fetcher).body.decision).toBe(forwarded)
  })

  it('passes a relaxation pause from the Python service through to the browser', async () => {
    const pause = { type: 'relaxation', constraint: 'yearRange', from: [1990, 1999], to: null }
    fakeAgentService({ reply: async () => jsonResponse(200, { ...pythonPayload, recommendations: [], pause }) })
    const { body } = await callRecommendationAgent({ ...turn, conversationId: 'conv-123' })
    expect((body as unknown as { pause?: unknown }).pause).toEqual(pause)
  })

  it('asks the Python service to forget a conversation', async () => {
    const fetcher = fakeAgentService({ url: 'https://agent.example.test/', reply: async () => ({ ok: true, status: 204, json: async () => ({}) }) })
    const captured = await callHandler('DELETE', { conversationId: 'conv-123' })
    expect(captured.status).toBe(200)
    expect(captured.body).toEqual({ deleted: true })
    const { url, init } = firstRequest(fetcher)
    expect(url).toBe('https://agent.example.test/agent/conversations/conv-123')
    expect(init.method).toBe('DELETE')
  })

  it('has nothing to forget without a Python service', async () => {
    const fetcher = fakeAgentService({ url: null })
    const { status, body } = await callHandler('DELETE', { conversationId: 'conv-123' })
    expect({ status, body }).toEqual({ status: 200, body: { deleted: false } })
    expect(fetcher).not.toHaveBeenCalled()
  })
})

describe('relaxation decisions when the Python service is down', () => {
  it('applies an accepted relaxation in the TypeScript fallback instead of losing it', async () => {
    process.env.AGENT_SERVICE_URL = 'https://agent.example.test'
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(503, {})))
    const preferences = { genres: ['Horror'], excludedGenres: [], platforms: [], moods: [], lastUpdatedAt: 1 }
    const answering = { type: 'relaxation', constraint: 'genres', from: ['Horror'], to: [] }
    const { body } = await callRecommendationAgent({ ...turn, message: 'Yes, loosen it', preferences, decision: 'accept', answering })
    const fallback = body as unknown as { preferences: { genres: string[] }; recommendations: unknown[] }
    expect(fallback.preferences.genres).toEqual([])
    expect(fallback.recommendations).toHaveLength(2)
  })

  it('forwards the proposal a decision answers', async () => {
    process.env.AGENT_SERVICE_URL = 'https://agent.example.test'
    const fetcher = vi.fn(async () => jsonResponse(200, pythonPayload))
    vi.stubGlobal('fetch', fetcher)
    const answering = { type: 'relaxation', constraint: 'maxRuntime', from: 90, to: 105 }
    await callRecommendationAgent({ ...turn, decision: 'reject', answering })
    const [, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit]
    expect(JSON.parse(init.body as string)).toMatchObject({ decision: 'reject', answering })
  })
})
