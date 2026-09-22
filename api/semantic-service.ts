import type { Recommendation } from '../src/lib/recommendation.js'
import { movieSemanticDocument, rerankWithSemanticMatches, rerankWithSemanticSimilarity, semanticSearch, type SemanticMatch, type SemanticRecommendation } from '../src/lib/semantic-retrieval.js'
import { embedTexts } from './embedding-client.js'
import { searchQdrant } from './qdrant-client.js'

export interface SemanticServiceResult {
  recommendations: SemanticRecommendation[]
  retrievalMode: 'qdrant' | 'embedding-memory' | 'local'
  embeddingMode: 'provider' | 'local'
  embeddingModel: string
  dimensions: number
  fallbackReason?: 'embedding-failed' | 'qdrant-failed'
}

interface SemanticServiceDependencies {
  embeddingFetcher?: Parameters<typeof embedTexts>[1]
  qdrantFetcher?: Parameters<typeof searchQdrant>[3]
}

const cosineSimilarity = (left: number[], right: number[]) => Math.max(0, left.reduce((total, value, index) => total + value * (right[index] ?? 0), 0))

const mergeLexicalEvidence = (matches: SemanticMatch[], lexicalMatches: SemanticMatch[]) => {
  const termsByMovie = new Map(lexicalMatches.map(({ movieId, matchedTerms }) => [movieId, matchedTerms]))
  return matches.map((match) => ({ ...match, matchedTerms: termsByMovie.get(match.movieId) ?? [] }))
}

export async function retrieveSemantically(
  query: string,
  recommendations: Recommendation[],
  dependencies: SemanticServiceDependencies = {},
): Promise<SemanticServiceResult> {
  const localResult = (): SemanticServiceResult => ({
    recommendations: rerankWithSemanticSimilarity(query, recommendations),
    retrievalMode: 'local',
    embeddingMode: 'local',
    embeddingModel: 'local-hash-v1',
    dimensions: 384,
  })
  if (recommendations.length === 0) return localResult()

  const documents = recommendations.map(({ movie }) => movieSemanticDocument(movie))
  let batch: Awaited<ReturnType<typeof embedTexts>>
  try {
    batch = await embedTexts([query, ...documents], dependencies.embeddingFetcher)
  } catch {
    return { ...localResult(), fallbackReason: 'embedding-failed' }
  }

  const [queryVector, ...candidateVectors] = batch.vectors
  const lexicalMatches = semanticSearch(query, recommendations.map(({ movie }) => movie))
  const memoryMatches = recommendations.map(({ movie }, index) => ({
    movieId: movie.id,
    score: cosineSimilarity(queryVector, candidateVectors[index]),
    matchedTerms: [],
  }))

  if (process.env.QDRANT_URL) {
    try {
      const qdrantMatches = await searchQdrant(
        queryVector,
        recommendations.map(({ movie }, index) => ({ movieId: movie.id, vector: candidateVectors[index] })),
        recommendations.length,
        dependencies.qdrantFetcher,
      )
      if (qdrantMatches) {
        return {
          recommendations: rerankWithSemanticMatches(recommendations, mergeLexicalEvidence(qdrantMatches, lexicalMatches)),
          retrievalMode: 'qdrant',
          embeddingMode: batch.mode,
          embeddingModel: batch.model,
          dimensions: batch.dimensions,
        }
      }
    } catch {
      return {
        recommendations: rerankWithSemanticMatches(recommendations, mergeLexicalEvidence(memoryMatches, lexicalMatches)),
        retrievalMode: batch.mode === 'provider' ? 'embedding-memory' : 'local',
        embeddingMode: batch.mode,
        embeddingModel: batch.model,
        dimensions: batch.dimensions,
        fallbackReason: 'qdrant-failed',
      }
    }
  }

  return {
    recommendations: rerankWithSemanticMatches(recommendations, mergeLexicalEvidence(memoryMatches, lexicalMatches)),
    retrievalMode: batch.mode === 'provider' ? 'embedding-memory' : 'local',
    embeddingMode: batch.mode,
    embeddingModel: batch.model,
    dimensions: batch.dimensions,
  }
}
