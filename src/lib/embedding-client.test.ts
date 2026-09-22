import { afterEach, describe, expect, it } from 'vitest'
import { embedTexts } from '../../api/embedding-client'

const originalEnvironment = {
  provider: process.env.EMBEDDING_PROVIDER,
  key: process.env.EMBEDDING_API_KEY,
  baseUrl: process.env.EMBEDDING_BASE_URL,
  model: process.env.EMBEDDING_MODEL,
  dimensions: process.env.EMBEDDING_DIMENSIONS,
}

afterEach(() => {
  const restore = (name: string, value: string | undefined) => {
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
  restore('EMBEDDING_API_KEY', originalEnvironment.key)
  restore('EMBEDDING_PROVIDER', originalEnvironment.provider)
  restore('EMBEDDING_BASE_URL', originalEnvironment.baseUrl)
  restore('EMBEDDING_MODEL', originalEnvironment.model)
  restore('EMBEDDING_DIMENSIONS', originalEnvironment.dimensions)
})

describe('embedding client', () => {
  it('uses deterministic local vectors when no provider is configured', async () => {
    delete process.env.EMBEDDING_BASE_URL
    delete process.env.EMBEDDING_MODEL
    delete process.env.EMBEDDING_PROVIDER

    const result = await embedTexts(['rainy Paris', 'space battle'])

    expect(result).toMatchObject({ mode: 'local', model: 'local-hash-v1', dimensions: 384 })
    expect(result.vectors).toHaveLength(2)
  })

  it('uses and validates an OpenAI-compatible embedding provider', async () => {
    delete process.env.EMBEDDING_PROVIDER
    process.env.EMBEDDING_API_KEY = 'test-key'
    process.env.EMBEDDING_BASE_URL = 'https://embeddings.example/v1/'
    process.env.EMBEDDING_MODEL = 'test-embedding'
    const fetcher = async (input: string, init: RequestInit) => {
      expect(input).toBe('https://embeddings.example/v1/embeddings')
      expect(init.headers).toMatchObject({ Authorization: 'Bearer test-key' })
      return {
        ok: true,
        status: 200,
        json: async () => ({ data: [{ index: 1, embedding: [0, 1] }, { index: 0, embedding: [1, 0] }] }),
      }
    }

    const result = await embedTexts(['first', 'second'], fetcher)

    expect(result).toEqual({ vectors: [[1, 0], [0, 1]], mode: 'provider', model: 'test-embedding', dimensions: 2 })
  })

  it('uses the native Gemini batch endpoint and retrieval prompts', async () => {
    process.env.EMBEDDING_PROVIDER = 'gemini'
    process.env.EMBEDDING_API_KEY = 'gemini-test-key'
    process.env.EMBEDDING_MODEL = 'gemini-embedding-2'
    process.env.EMBEDDING_DIMENSIONS = '768'
    delete process.env.EMBEDDING_BASE_URL
    const fetcher = async (input: string, init: RequestInit) => {
      expect(input).toBe('https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents')
      expect(init.headers).toMatchObject({ 'x-goog-api-key': 'gemini-test-key' })
      const body = JSON.parse(String(init.body))
      expect(body.requests[0].content.parts[0].text).toBe('task: search result | query: rainy Paris')
      expect(body.requests[1].content.parts[0].text).toBe('title: none | text: autumn cafes')
      expect(body.requests[0].embedContentConfig.outputDimensionality).toBe(768)
      return {
        ok: true,
        status: 200,
        json: async () => ({ embeddings: [{ values: [1, 0] }, { values: [0.9, 0.1] }] }),
      }
    }

    const result = await embedTexts(['rainy Paris', 'autumn cafes'], fetcher)

    expect(result).toEqual({ vectors: [[1, 0], [0.9, 0.1]], mode: 'provider', model: 'gemini-embedding-2', dimensions: 2 })
  })
})
