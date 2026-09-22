import { afterEach, describe, expect, it } from 'vitest'
import { embedTexts } from '../../api/embedding-client'

const originalEnvironment = {
  key: process.env.EMBEDDING_API_KEY,
  baseUrl: process.env.EMBEDDING_BASE_URL,
  model: process.env.EMBEDDING_MODEL,
}

afterEach(() => {
  const restore = (name: string, value: string | undefined) => {
    if (value === undefined) delete process.env[name]
    else process.env[name] = value
  }
  restore('EMBEDDING_API_KEY', originalEnvironment.key)
  restore('EMBEDDING_BASE_URL', originalEnvironment.baseUrl)
  restore('EMBEDDING_MODEL', originalEnvironment.model)
})

describe('embedding client', () => {
  it('uses deterministic local vectors when no provider is configured', async () => {
    delete process.env.EMBEDDING_BASE_URL
    delete process.env.EMBEDDING_MODEL

    const result = await embedTexts(['rainy Paris', 'space battle'])

    expect(result).toMatchObject({ mode: 'local', model: 'local-hash-v1', dimensions: 384 })
    expect(result.vectors).toHaveLength(2)
  })

  it('uses and validates an OpenAI-compatible embedding provider', async () => {
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
})
