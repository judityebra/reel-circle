import { afterEach, describe, expect, it, vi } from 'vitest'
import handler from '../../api/agent'
import { deleteAgentConversation, requestAgentTurn, type RelaxationPause } from './agent-api'
import { createConversationPreferences } from './constraints'
import type { Movie } from './recommendation'

const candidates: Movie[] = [
  { id: 'horror', title: 'The Fright', year: 2020, runtime: 85, genres: ['Horror'], platforms: [], rating: 4.8, moods: ['Twisty'] },
  { id: 'mystery', title: 'The Puzzle', year: 2021, runtime: 88, genres: ['Mystery'], platforms: [], rating: 4.2, moods: ['Twisty'] },
  { id: 'long', title: 'Long Mystery', year: 2022, runtime: 140, genres: ['Mystery'], platforms: [], rating: 4.9, moods: ['Twisty'] },
]

const originalApiKey = process.env.OPENAI_API_KEY

/** A fetcher that answers like the agent service, plus `extra` fields, and remembers what was sent. */
function recordingFetcher(extra: Record<string, unknown> = {}) {
  let body = ''
  const fetcher = async (_input: string, init: RequestInit) => {
    body = String(init.body)
    return {
      ok: true,
      status: 200,
      json: async () => ({ preferences: createConversationPreferences(), recommendations: [], meta: { extractionMode: 'deterministic', durationMs: 1, trace: [] }, ...extra }),
    }
  }
  return { fetcher, sentBody: () => body }
}

afterEach(() => {
  vi.unstubAllGlobals()
  if (originalApiKey === undefined) delete process.env.OPENAI_API_KEY
  else process.env.OPENAI_API_KEY = originalApiKey
})

describe('agent API', () => {
  it('uses deterministic extraction and validates hard constraints without a model key', async () => {
    delete process.env.OPENAI_API_KEY
    let statusCode = 0
    let payload: unknown
    const response = {
      status(code: number) { statusCode = code; return this },
      json(body: unknown) { payload = body },
      setHeader() {},
    }

    await handler({
      method: 'POST',
      body: {
        message: 'Something darker, under 90 minutes',
        preferences: { ...createConversationPreferences(1), excludedGenres: ['Horror'] },
        candidates,
      },
    }, response)

    expect(statusCode).toBe(200)
    expect(payload).toMatchObject({
      preferences: { excludedGenres: ['Horror'], maxRuntime: 90, moods: ['Twisty'] },
      recommendations: [{ movie: { id: 'mystery' } }],
      meta: {
        extractionMode: 'deterministic',
        semantic: { retrievalMode: 'local', embeddingMode: 'local', embeddingModel: 'local-hash-v1', dimensions: 384 },
        trace: [
          { tool: 'extract_preferences', status: 'completed' },
          { tool: 'filter_candidates', status: 'completed' },
          { tool: 'semantic_retrieval', status: 'completed' },
          { tool: 'rank_candidates', status: 'completed' },
          { tool: 'validate_results', status: 'completed', observation: '0 hard-constraint violations' },
        ],
      },
    })
  })

  it('rejects malformed requests', async () => {
    let statusCode = 0
    const response = {
      status(code: number) { statusCode = code; return this },
      json() {},
      setHeader() {},
    }

    await handler({ method: 'POST', body: { message: '' } }, response)

    expect(statusCode).toBe(400)
  })

  it('retries with JSON Object Mode when strict schemas are unsupported', async () => {
    process.env.OPENAI_API_KEY = 'test-key'
    const modelFetch = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 400 })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: JSON.stringify({ genres: ['Mystery'], excludedGenres: ['Horror'], platforms: [], moods: ['Twisty'], maxRuntime: 100, yearRange: null, languages: [], peopleCount: null, viewingContext: null, clarificationQuestion: null }) } }] }),
      })
    vi.stubGlobal('fetch', modelFetch)
    let payload: { meta?: { extractionMode?: string }; recommendations?: Array<{ movie: Movie }> } = {}
    const response = {
      status() { return this },
      json(body: unknown) { payload = body as typeof payload },
      setHeader() {},
    }

    await handler({ method: 'POST', body: { message: 'Dark mystery, no horror', candidates } }, response)

    expect(modelFetch).toHaveBeenCalledTimes(2)
    expect(JSON.parse(modelFetch.mock.calls[1][1].body).response_format).toEqual({ type: 'json_object' })
    expect(payload.meta?.extractionMode).toBe('model')
    expect(payload.recommendations?.map(({ movie }) => movie.id)).toEqual(['mystery'])
    expect(payload.recommendations?.[0]).toMatchObject({ semanticScore: expect.any(Number), semanticTerms: expect.any(Array) })
  })

  it('strips raw ratings and watch history from client requests', async () => {
    const { fetcher, sentBody } = recordingFetcher()
    const privateProfile = {
      id: 'private', name: 'Private', color: '#000', preferredGenres: ['Drama'],
      ratings: [{ title: 'Secret rating', rating: 5 }], watched: [{ title: 'Secret watch' }],
    }

    await requestAgentTurn('/api', 'Something gentle', createConversationPreferences(), candidates, [privateProfile], { fetcher })

    expect(sentBody()).not.toContain('Secret rating')
    expect(sentBody()).not.toContain('Secret watch')
    expect(JSON.parse(sentBody()).profiles[0]).toMatchObject({ ratings: [], watched: [] })
  })

  it('sends the conversation id so the server remembers the conversation', async () => {
    const { fetcher, sentBody } = recordingFetcher({ conversationId: 'conv-1' })
    const result = await requestAgentTurn('/api', 'Something gentle', createConversationPreferences(), candidates, [], { conversationId: 'conv-1', fetcher })
    expect(JSON.parse(sentBody()).conversationId).toBe('conv-1')
    expect(result.conversationId).toBe('conv-1')
  })

  it('answers a relaxation proposal with the user decision and returns the next pause', async () => {
    const answering: RelaxationPause = { type: 'relaxation', constraint: 'yearRange', from: [1990, 1999], to: null }
    const pause = { type: 'relaxation', constraint: 'maxRuntime', from: 90, to: 105 }
    const { fetcher, sentBody } = recordingFetcher({ conversationId: 'conv-1', pause })
    const result = await requestAgentTurn('/api', 'No, keep the year', createConversationPreferences(), candidates, [], { conversationId: 'conv-1', answer: { decision: 'reject', answering }, fetcher })
    expect(JSON.parse(sentBody())).toMatchObject({ decision: 'reject', answering })
    expect(result.pause).toEqual(pause)
  })

  it('asks the server to forget a conversation', async () => {
    const calls: Array<[string, RequestInit]> = []
    const fetcher = async (input: string, init: RequestInit) => {
      calls.push([input, init])
      return { ok: true, status: 200, json: async () => ({ deleted: true }) }
    }
    await deleteAgentConversation('/api/', 'conv-1', fetcher)
    expect(calls[0][0]).toBe('/api/agent')
    expect(calls[0][1].method).toBe('DELETE')
    expect(JSON.parse(String(calls[0][1].body))).toEqual({ conversationId: 'conv-1' })
  })

  it('reports a conversation the server could not forget', async () => {
    const fetcher = async () => ({ ok: false, status: 502, json: async () => ({}) })
    await expect(deleteAgentConversation('/api', 'conv-1', fetcher)).rejects.toThrow('502')
  })

  it('rejects an answer whose pause the browser could not render', async () => {
    const fetcher = async () => ({
      ok: true,
      status: 200,
      json: async () => ({ preferences: createConversationPreferences(), recommendations: [], pause: { type: 'relaxation', constraint: 'mood' }, meta: { extractionMode: 'deterministic', durationMs: 1, trace: [] } }),
    })
    await expect(requestAgentTurn('/api', 'Anything', createConversationPreferences(), candidates, [], { fetcher })).rejects.toThrow('invalid data')
  })
})
