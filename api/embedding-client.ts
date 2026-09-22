import { createSemanticVector } from '../src/lib/semantic-retrieval.js'

export interface EmbeddingBatch {
  vectors: number[][]
  mode: 'provider' | 'local'
  model: string
  dimensions: number
}

type Fetcher = (input: string, init: RequestInit) => Promise<Pick<Response, 'ok' | 'status' | 'json'>>

export async function embedTexts(texts: string[], fetcher: Fetcher = fetch): Promise<EmbeddingBatch> {
  const provider = process.env.EMBEDDING_PROVIDER?.toLowerCase()
  const baseUrl = process.env.EMBEDDING_BASE_URL?.replace(/\/$/, '')
  const model = process.env.EMBEDDING_MODEL
  if (!model || (provider !== 'gemini' && !baseUrl)) {
    const vectors = texts.map(createSemanticVector)
    return { vectors, mode: 'local', model: 'local-hash-v1', dimensions: vectors[0]?.length ?? 384 }
  }

  const dimensions = Math.max(128, Math.min(3_072, Number(process.env.EMBEDDING_DIMENSIONS) || 768))
  const response = provider === 'gemini'
    ? await fetcher(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:batchEmbedContents`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(process.env.EMBEDDING_API_KEY ? { 'x-goog-api-key': process.env.EMBEDDING_API_KEY } : {}),
      },
      body: JSON.stringify({
        requests: texts.map((text, index) => ({
          model: `models/${model}`,
          content: { parts: [{ text: index === 0 ? `task: search result | query: ${text}` : `title: none | text: ${text}` }] },
          embedContentConfig: { outputDimensionality: dimensions, autoTruncate: true },
        })),
      }),
    })
    : await fetcher(`${baseUrl}/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(process.env.EMBEDDING_API_KEY ? { Authorization: `Bearer ${process.env.EMBEDDING_API_KEY}` } : {}),
    },
    body: JSON.stringify({ model, input: texts, encoding_format: 'float' }),
  })
  if (!response.ok) throw new Error(`Embedding request failed: ${response.status}`)
  const payload = await response.json() as {
    data?: Array<{ index?: number; embedding?: unknown }>
    embeddings?: Array<{ values?: unknown }>
  }
  const vectors = provider === 'gemini'
    ? (payload.embeddings ?? []).map(({ values }) => values)
    : [...(payload.data ?? [])].sort((left, right) => (left.index ?? 0) - (right.index ?? 0)).map(({ embedding }) => embedding)
  if (vectors.length !== texts.length || vectors.some((vector) => !Array.isArray(vector) || vector.length === 0 || vector.some((value) => typeof value !== 'number' || !Number.isFinite(value)))) {
    throw new Error('Embedding provider returned invalid vectors')
  }
  const numericVectors = vectors as number[][]
  const returnedDimensions = numericVectors[0].length
  if (numericVectors.some((vector) => vector.length !== returnedDimensions)) throw new Error('Embedding provider returned inconsistent dimensions')
  return { vectors: numericVectors, mode: 'provider', model, dimensions: returnedDimensions }
}
