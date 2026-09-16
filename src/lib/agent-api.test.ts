import { afterEach, describe, expect, it, vi } from 'vitest'
import handler from '../../api/agent'
import { requestAgentTurn } from './agent-api'
import { createConversationPreferences } from './constraints'
import type { Movie } from './recommendation'

const candidates: Movie[] = [
  { id: 'horror', title: 'The Fright', year: 2020, runtime: 85, genres: ['Horror'], platforms: [], rating: 4.8, moods: ['Twisty'] },
  { id: 'mystery', title: 'The Puzzle', year: 2021, runtime: 88, genres: ['Mystery'], platforms: [], rating: 4.2, moods: ['Twisty'] },
  { id: 'long', title: 'Long Mystery', year: 2022, runtime: 140, genres: ['Mystery'], platforms: [], rating: 4.9, moods: ['Twisty'] },
]

const originalApiKey = process.env.OPENAI_API_KEY

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
        trace: [
          { tool: 'extract_preferences', status: 'completed' },
          { tool: 'filter_candidates', status: 'completed' },
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
  })

  it('strips raw ratings and watch history from client requests', async () => {
    let requestBody = ''
    const fetcher = async (_input: string, init: RequestInit) => {
      requestBody = String(init.body)
      return {
        ok: true,
        status: 200,
        json: async () => ({ preferences: createConversationPreferences(), recommendations: [], meta: { extractionMode: 'deterministic', durationMs: 1, trace: [] } }),
      }
    }
    const privateProfile = {
      id: 'private', name: 'Private', color: '#000', preferredGenres: ['Drama'],
      ratings: [{ title: 'Secret rating', rating: 5 }], watched: [{ title: 'Secret watch' }],
    }

    await requestAgentTurn('/api', 'Something gentle', createConversationPreferences(), candidates, [privateProfile], fetcher)

    expect(requestBody).not.toContain('Secret rating')
    expect(requestBody).not.toContain('Secret watch')
    expect(JSON.parse(requestBody).profiles[0]).toMatchObject({ ratings: [], watched: [] })
  })
})
