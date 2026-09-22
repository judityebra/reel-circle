import type { Movie, Recommendation } from './recommendation'

export interface SemanticMatch {
  movieId: string
  score: number
  matchedTerms: string[]
}

export interface SemanticRecommendation extends Recommendation {
  semanticScore: number
  semanticTerms: string[]
}

const dimensions = 384
const stopWords = new Set(['a', 'an', 'and', 'but', 'for', 'i', 'in', 'is', 'it', 'me', 'movie', 'of', 'on', 'or', 'something', 'the', 'to', 'want', 'with'])
const concepts: Record<string, string[]> = {
  autumn: ['autumn', 'fall', 'leaves', 'harvest', 'cozy', 'nostalgic', 'warm'],
  beautiful: ['beautiful', 'visual', 'cinematography', 'lush', 'elegant', 'stunning'],
  comfort: ['comfort', 'cozy', 'gentle', 'warm', 'hopeful', 'uplifting'],
  feminine: ['feminine', 'female', 'woman', 'women', 'girl', 'stylish'],
  paris: ['paris', 'french', 'france', 'romance', 'city'],
  rainy: ['rainy', 'rain', 'melancholy', 'quiet', 'cozy', 'atmospheric'],
  mysterious: ['mysterious', 'mystery', 'enigmatic', 'suspense', 'twisty'],
  elegant: ['elegant', 'refined', 'stylish', 'beautiful', 'lush'],
}

const tokenize = (value: string) => value
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .toLowerCase()
  .match(/[a-z0-9]+/g)
  ?.filter((token) => token.length > 1 && !stopWords.has(token)) ?? []

const hash = (value: string) => {
  let result = 2166136261
  for (const character of value) {
    result ^= character.charCodeAt(0)
    result = Math.imul(result, 16777619)
  }
  return Math.abs(result) % dimensions
}

const expandTokens = (tokens: string[]) => [...new Set(tokens.flatMap((token) => [token, ...(concepts[token] ?? [])]))]

export function movieSemanticDocument(movie: Movie): string {
  return [
    movie.title,
    movie.overview,
    movie.genres.join(' '),
    movie.keywords?.join(' '),
    movie.cast?.join(' '),
    movie.director,
    movie.tone,
    movie.moods?.join(' '),
    movie.language,
  ].filter(Boolean).join(' ')
}

export function createSemanticVector(value: string): number[] {
  const vector = Array<number>(dimensions).fill(0)
  const tokens = expandTokens(tokenize(value))
  tokens.forEach((token) => { vector[hash(token)] += 1 })
  const magnitude = Math.sqrt(vector.reduce((total, item) => total + item * item, 0)) || 1
  return vector.map((item) => item / magnitude)
}

const cosineSimilarity = (left: number[], right: number[]) => left.reduce((total, value, index) => total + value * (right[index] ?? 0), 0)

export function semanticSearch(query: string, movies: Movie[], limit = movies.length): SemanticMatch[] {
  const queryTerms = expandTokens(tokenize(query))
  const queryVector = createSemanticVector(query)
  return movies
    .map((movie) => {
      const document = movieSemanticDocument(movie)
      const documentTokens = new Set(expandTokens(tokenize(document)))
      const matchedTerms = queryTerms.filter((term) => documentTokens.has(term))
      return { movieId: movie.id, score: Math.max(0, cosineSimilarity(queryVector, createSemanticVector(document))), matchedTerms }
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, Math.max(0, limit))
}

export function rerankWithSemanticMatches(recommendations: Recommendation[], semanticMatches: SemanticMatch[]): SemanticRecommendation[] {
  const matches = new Map(semanticMatches.map((match) => [match.movieId, match]))
  return recommendations
    .map((recommendation) => {
      const match = matches.get(recommendation.movie.id) ?? { score: 0, matchedTerms: [] }
      const semanticScore = Math.round(match.score * 100)
      return {
        ...recommendation,
        score: Math.round(recommendation.score * 0.72 + semanticScore * 0.28),
        semanticScore,
        semanticTerms: match.matchedTerms.slice(0, 5),
      }
    })
    .sort((left, right) => right.score - left.score)
}

export function rerankWithSemanticSimilarity(query: string, recommendations: Recommendation[]): SemanticRecommendation[] {
  return rerankWithSemanticMatches(recommendations, semanticSearch(query, recommendations.map(({ movie }) => movie)))
}
