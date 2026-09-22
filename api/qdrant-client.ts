import { createHash } from 'node:crypto'
import type { SemanticMatch } from '../src/lib/semantic-retrieval.js'

interface VectorCandidate {
  movieId: string
  vector: number[]
}

type Fetcher = (input: string, init: RequestInit) => Promise<Pick<Response, 'ok' | 'status' | 'json'>>

const pointId = (movieId: string) => {
  const hex = createHash('sha256').update(movieId).digest('hex').slice(0, 32)
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20)}`
}

export async function searchQdrant(
  queryVector: number[],
  candidates: VectorCandidate[],
  limit: number,
  fetcher: Fetcher = fetch,
): Promise<SemanticMatch[] | undefined> {
  const baseUrl = process.env.QDRANT_URL?.replace(/\/$/, '')
  if (!baseUrl || candidates.length === 0) return undefined
  const collection = process.env.QDRANT_COLLECTION ?? `reel-circle-${queryVector.length}`
  const headers = {
    'Content-Type': 'application/json',
    ...(process.env.QDRANT_API_KEY ? { 'api-key': process.env.QDRANT_API_KEY } : {}),
  }
  const collectionUrl = `${baseUrl}/collections/${encodeURIComponent(collection)}`
  const existing = await fetcher(collectionUrl, { method: 'GET', headers })
  if (existing.status === 404) {
    const created = await fetcher(collectionUrl, {
      method: 'PUT',
      headers,
      body: JSON.stringify({ vectors: { size: queryVector.length, distance: 'Cosine' } }),
    })
    if (!created.ok) throw new Error(`Qdrant collection creation failed: ${created.status}`)
  } else if (!existing.ok) {
    throw new Error(`Qdrant collection lookup failed: ${existing.status}`)
  }

  const points = candidates.map(({ movieId, vector }) => ({ id: pointId(movieId), vector, payload: { movieId } }))
  const upserted = await fetcher(`${collectionUrl}/points?wait=true`, {
    method: 'PUT',
    headers,
    body: JSON.stringify({ points }),
  })
  if (!upserted.ok) throw new Error(`Qdrant upsert failed: ${upserted.status}`)

  const queryBody = {
    query: queryVector,
    filter: { must: [{ has_id: candidates.map(({ movieId }) => pointId(movieId)) }] },
    limit: Math.max(1, Math.min(limit, candidates.length)),
    with_payload: true,
    with_vector: false,
  }
  let searched = await fetcher(`${collectionUrl}/points/query`, {
    method: 'POST',
    headers,
    body: JSON.stringify(queryBody),
  })
  let legacy = false
  if (searched.status === 404) {
    legacy = true
    searched = await fetcher(`${collectionUrl}/points/search`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...queryBody, vector: queryVector, query: undefined }),
    })
  }
  if (!searched.ok) throw new Error(`Qdrant search failed: ${searched.status}`)
  const payload = await searched.json() as { result?: { points?: Array<{ score?: number; payload?: { movieId?: string } }> } | Array<{ score?: number; payload?: { movieId?: string } }> }
  const results = legacy ? payload.result : !Array.isArray(payload.result) ? payload.result?.points : []
  return (Array.isArray(results) ? results : []).flatMap(({ score, payload: itemPayload }) => itemPayload?.movieId
    ? [{ movieId: itemPayload.movieId, score: Math.max(0, Math.min(1, score ?? 0)), matchedTerms: [] }]
    : [])
}
