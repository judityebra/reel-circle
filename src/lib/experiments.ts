import { getMovieFeatures, type Movie, type Profile, type Recommendation } from './recommendation'

export interface RankingComparison {
  baselineTop?: string
  modelTop?: string
  topChanged: boolean
  topThreeOverlap: number
  meanRankShift: number
}

export interface RecommendationPath {
  label: string
  kind: string
  weight: number
  path: [string, string, string]
}

export function compareRankings(
  baseline: Recommendation[],
  model: Recommendation[],
): RankingComparison {
  const baselineTop = baseline[0]?.movie.id
  const modelTop = model[0]?.movie.id
  const baselineThree = baseline.slice(0, 3).map(({ movie }) => movie.id)
  const modelThree = model.slice(0, 3).map(({ movie }) => movie.id)
  const overlap = baselineThree.filter((id) => modelThree.includes(id))
  const sharedIds = baseline.map(({ movie }) => movie.id).filter((id) => model.some(({ movie }) => movie.id === id))
  const rankShift = sharedIds.length === 0 ? 0 : sharedIds.reduce((total, id) => {
    const baselineRank = baseline.findIndex(({ movie }) => movie.id === id)
    const modelRank = model.findIndex(({ movie }) => movie.id === id)
    return total + Math.abs(baselineRank - modelRank)
  }, 0) / sharedIds.length

  return {
    baselineTop,
    modelTop,
    topChanged: baselineTop !== modelTop,
    topThreeOverlap: Math.round((overlap.length / Math.max(1, Math.min(3, baseline.length, model.length))) * 100),
    meanRankShift: Number(rankShift.toFixed(1)),
  }
}

export function getDisagreement(recommendation: Recommendation): number {
  const scores = recommendation.matchByProfile.map(({ score }) => score)
  return scores.length < 2 ? 0 : Math.max(...scores) - Math.min(...scores)
}

export function explainRecommendationPath(profile: Profile, movie: Movie): RecommendationPath | undefined {
  const signals = [
    ...movie.genres.map((genre) => ({ kind: 'genre', label: genre, key: genre, weight: profile.tasteWeights?.[genre] ?? (profile.preferredGenres.includes(genre) ? 0.25 : 0) })),
    ...getMovieFeatures(movie).map((feature) => {
      const [kind, ...labelParts] = feature.split(':')
      return { kind, label: labelParts.join(':'), key: feature, weight: profile.featureWeights?.[feature] ?? 0 }
    }),
  ].filter(({ weight }) => weight !== 0)
  const strongest = signals.sort((left, right) => Math.abs(right.weight) - Math.abs(left.weight))[0]
  if (!strongest) return undefined
  return {
    label: strongest.label,
    kind: strongest.kind,
    weight: strongest.weight,
    path: [profile.name, `${strongest.kind}: ${strongest.label}`, movie.title],
  }
}

export interface ExperimentRecord {
  id: string
  timestamp: number
  profileId: string
  model: 'MLP' | 'GCN'
  examples: number
  confidence: number
  loss: number
  topChanged: boolean
  topThreeOverlap: number
  meanRankShift: number
}