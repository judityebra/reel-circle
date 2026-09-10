import type { Tensor2D } from '@tensorflow/tfjs'
import { encodeMovie, type NeuralExample } from './neural'
import { getMovieFeatures, type Movie } from './recommendation'

const minimumExamples = 4

export interface MovieFeatureGraph {
  nodeIds: string[]
  movieIndices: number[]
  features: number[][]
  adjacency: number[][]
}

export interface GraphNeuralResult {
  status: 'ready' | 'insufficient-data' | 'error'
  confidence: number
  predictions: Record<string, number>
  embeddings: Record<string, number[]>
  loss?: number
  examples?: number
  graphStats?: { movieNodes: number; featureNodes: number; edges: number; layers: 2 }
}

interface TrainingOptions {
  epochs?: number
}

const graphFeatures = (movie: Movie) => [
  ...movie.genres.map((genre) => `genre:${genre}`),
  ...movie.platforms.map((platform) => `platform:${platform}`),
  ...getMovieFeatures(movie),
]

const featureVector = (feature: string) => {
  const vector = Array<number>(34).fill(0)
  let hash = 2166136261
  for (let index = 0; index < feature.length; index += 1) {
    hash ^= feature.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  vector[Math.abs(hash) % 32] = 1
  return vector
}

export function buildMovieFeatureGraph(movies: Movie[]): MovieFeatureGraph {
  const features = [...new Set(movies.flatMap(graphFeatures))]
  const nodeIds = [
    ...movies.map(({ id }) => `movie:${id}`),
    ...features.map((feature) => `feature:${feature}`),
  ]
  const nodeIndex = new Map(nodeIds.map((id, index) => [id, index]))
  const adjacency: number[][] = Array.from({ length: nodeIds.length }, (_, row) =>
    Array.from({ length: nodeIds.length }, (_, column) => row === column ? 1 : 0),
  )
  movies.forEach((movie) => {
    const movieIndex = nodeIndex.get(`movie:${movie.id}`)!
    graphFeatures(movie).forEach((feature) => {
      const featureIndex = nodeIndex.get(`feature:${feature}`)!
      adjacency[movieIndex][featureIndex] = 1
      adjacency[featureIndex][movieIndex] = 1
    })
  })
  const degrees = adjacency.map((row) => row.reduce<number>((total, value) => total + value, 0))
  const normalized = adjacency.map((row, rowIndex) => row.map((value, columnIndex) =>
    value / Math.sqrt(degrees[rowIndex] * degrees[columnIndex]),
  ))

  return {
    nodeIds,
    movieIndices: movies.map((_, index) => index),
    features: [...movies.map(encodeMovie), ...features.map(featureVector)],
    adjacency: normalized,
  }
}

export async function trainGraphNeuralTaste(
  movies: Movie[],
  examples: NeuralExample[],
  options: TrainingOptions = {},
): Promise<GraphNeuralResult> {
  const movieIndexById = new Map(movies.map((movie, index) => [movie.id, index]))
  const usableExamples = examples.filter(({ movieId }) => movieIndexById.has(movieId))
  if (usableExamples.length < minimumExamples) {
    return { status: 'insufficient-data', confidence: 0, predictions: {}, embeddings: {} }
  }

  const tf = await import('@tensorflow/tfjs')
  await tf.setBackend('cpu')
  await tf.ready()
  const graph = buildMovieFeatureGraph(movies)
  const adjacency = tf.tensor2d(graph.adjacency)
  const nodeFeatures = tf.tensor2d(graph.features)
  const labeledIndices = tf.tensor1d(usableExamples.map(({ movieId }) => movieIndexById.get(movieId)!), 'int32')
  const targets = tf.tensor2d(usableExamples.map(({ preference }) => [Math.max(0, Math.min(1, preference))]))
  const firstWeights = tf.variable(tf.initializers.glorotUniform({ seed: 41 }).apply([34, 12]) as Tensor2D)
  const secondWeights = tf.variable(tf.initializers.glorotUniform({ seed: 43 }).apply([12, 6]) as Tensor2D)
  const outputWeights = tf.variable(tf.initializers.glorotUniform({ seed: 47 }).apply([6, 1]) as Tensor2D)
  const optimizer = tf.train.adam(0.025)

  const forward = () => tf.tidy(() => {
    const firstLayer = adjacency.matMul(nodeFeatures).matMul(firstWeights).relu()
    const secondLayer = adjacency.matMul(firstLayer).matMul(secondWeights).relu()
    const output = secondLayer.matMul(outputWeights).sigmoid()
    return { firstLayer: tf.keep(firstLayer), output: tf.keep(output) }
  })

  try {
    let lossValue = 1
    for (let epoch = 0; epoch < (options.epochs ?? 35); epoch += 1) {
      const loss = optimizer.minimize(() => tf.tidy(() => {
        const firstLayer = adjacency.matMul(nodeFeatures).matMul(firstWeights).relu()
        const secondLayer = adjacency.matMul(firstLayer).matMul(secondWeights).relu()
        const predictions = secondLayer.matMul(outputWeights).sigmoid().gather(labeledIndices)
        return tf.losses.meanSquaredError(targets, predictions).mean()
      }), true)
      if (loss) {
        lossValue = (await loss.data())[0]
        loss.dispose()
      }
    }

    const { firstLayer, output } = forward()
    const outputValues = await output.data()
    const embeddingValues = await firstLayer.array() as number[][]
    firstLayer.dispose()
    output.dispose()
    const predictions = Object.fromEntries(movies.map((movie, index) => [movie.id, Math.round(outputValues[index] * 100)]))
    const embeddings = Object.fromEntries(movies.map((movie, index) => [movie.id, embeddingValues[index]]))
    const dataConfidence = Math.min(1, usableExamples.length / 14)
    const fitConfidence = Math.max(0, 1 - Math.min(lossValue, 1))
    const confidence = Math.round(dataConfidence * fitConfidence * 100)
    const edgeCount = movies.reduce((total, movie) => total + graphFeatures(movie).length, 0)
    return {
      status: 'ready',
      confidence,
      predictions,
      embeddings,
      loss: lossValue,
      examples: usableExamples.length,
      graphStats: { movieNodes: movies.length, featureNodes: graph.nodeIds.length - movies.length, edges: edgeCount, layers: 2 },
    }
  } catch {
    return { status: 'error', confidence: 0, predictions: {}, embeddings: {} }
  } finally {
    adjacency.dispose()
    nodeFeatures.dispose()
    labeledIndices.dispose()
    targets.dispose()
    firstWeights.dispose()
    secondWeights.dispose()
    outputWeights.dispose()
    optimizer.dispose()
  }
}