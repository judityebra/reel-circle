/**
 * Ranking and summary metrics used by the offline evaluations.
 * Pure functions so they can be unit tested independently of the benchmarks.
 */

/** NDCG@k with linear gains. `ranked` holds the true relevance of each item in predicted order. */
export function ndcgAtK(ranked: number[], allRelevances: number[], k: number): number {
  const dcg = (values: number[]) => values
    .slice(0, k)
    .reduce((total, relevance, index) => total + relevance / Math.log2(index + 2), 0)
  const ideal = dcg([...allRelevances].sort((left, right) => right - left))
  return ideal === 0 ? 0 : dcg(ranked) / ideal
}

/** Share of the top-k predicted items whose true value meets the threshold. */
export function precisionAtK(ranked: number[], k: number, threshold: number): number {
  const top = ranked.slice(0, k)
  if (top.length === 0) return 0
  return top.filter((value) => value >= threshold).length / top.length
}

/** 1-based ranks with ties sharing their average rank. */
export function averageRanks(values: number[]): number[] {
  const order = values.map((value, index) => ({ value, index })).sort((left, right) => left.value - right.value)
  const ranks = Array<number>(values.length)
  let start = 0
  while (start < order.length) {
    let end = start
    while (end + 1 < order.length && order[end + 1].value === order[start].value) end += 1
    const rank = (start + end) / 2 + 1
    for (let position = start; position <= end; position += 1) ranks[order[position].index] = rank
    start = end + 1
  }
  return ranks
}

export function pearson(left: number[], right: number[]): number {
  if (left.length !== right.length || left.length < 2) return 0
  const meanLeft = mean(left)
  const meanRight = mean(right)
  let covariance = 0
  let varianceLeft = 0
  let varianceRight = 0
  for (let index = 0; index < left.length; index += 1) {
    const deltaLeft = left[index] - meanLeft
    const deltaRight = right[index] - meanRight
    covariance += deltaLeft * deltaRight
    varianceLeft += deltaLeft ** 2
    varianceRight += deltaRight ** 2
  }
  // A constant ranking carries no ordering information.
  if (varianceLeft === 0 || varianceRight === 0) return 0
  return covariance / Math.sqrt(varianceLeft * varianceRight)
}

/** Spearman rank correlation (Pearson on tie-averaged ranks). Constant inputs score 0. */
export function spearman(predicted: number[], actual: number[]): number {
  return pearson(averageRanks(predicted), averageRanks(actual))
}

export function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((total, value) => total + value, 0) / values.length
}

export interface Interval {
  mean: number
  low: number
  high: number
}

/** Percentile bootstrap confidence interval for the mean. */
export function bootstrapCI(values: number[], rng: () => number, resamples = 1000, level = 0.95): Interval {
  if (values.length === 0) return { mean: 0, low: 0, high: 0 }
  const means: number[] = []
  for (let sample = 0; sample < resamples; sample += 1) {
    let total = 0
    for (let draw = 0; draw < values.length; draw += 1) total += values[Math.floor(rng() * values.length)]
    means.push(total / values.length)
  }
  means.sort((left, right) => left - right)
  const tail = (1 - level) / 2
  const at = (quantile: number) => means[Math.min(means.length - 1, Math.max(0, Math.floor(quantile * means.length)))]
  return { mean: mean(values), low: at(tail), high: at(1 - tail) }
}

export function formatInterval({ mean: value, low, high }: Interval, digits = 3): string {
  return `${value.toFixed(digits)} [${low.toFixed(digits)}, ${high.toFixed(digits)}]`
}

/** Set equality for string lists, ignoring order and duplicates. */
export function sameSet(left: readonly string[] = [], right: readonly string[] = []): boolean {
  const leftSet = new Set(left)
  const rightSet = new Set(right)
  return leftSet.size === rightSet.size && [...leftSet].every((item) => rightSet.has(item))
}

/** Recall of expected items; undefined when nothing was expected. */
export function recall(expected: readonly string[], predicted: readonly string[]): { hits: number; total: number } {
  const predictedSet = new Set(predicted)
  return { hits: expected.filter((item) => predictedSet.has(item)).length, total: expected.length }
}
