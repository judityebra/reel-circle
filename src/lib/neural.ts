import type { Tensor } from '@tensorflow/tfjs'
import { getMovieFeatures, type Movie, type Profile } from './recommendation'
import type { FeedbackEvent } from './session'

const featureBuckets = 32
const inputSize = featureBuckets + 2
const minimumExamples = 4

export interface NeuralExample {
  movieId: string
  preference: number
}

export interface NeuralTasteResult {
  status: 'ready' | 'insufficient-data' | 'error'
  confidence: number
  predictions: Record<string, number>
  activations?: Record<string, { hiddenOne: number[]; hiddenTwo: number[] }>
  loss?: number
  examples?: number
}

interface TrainingOptions {
  epochs?: number
}

const hashFeature = (value: string) => {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return Math.abs(hash) % featureBuckets
}

export function encodeMovie(movie: Movie): number[] {
  const vector = Array<number>(inputSize).fill(0)
  const categoricalFeatures = [
    ...movie.genres.map((genre) => `genre:${genre}`),
    ...getMovieFeatures(movie),
  ]
  categoricalFeatures.forEach((feature) => {
    vector[hashFeature(feature)] = Math.min(1, vector[hashFeature(feature)] + 0.5)
  })
  vector[featureBuckets] = Math.min(movie.runtime / 240, 1)
  vector[featureBuckets + 1] = Math.min(movie.rating / 5, 1)
  return vector
}

export async function trainNeuralTaste(
  movies: Movie[],
  examples: NeuralExample[],
  options: TrainingOptions = {},
): Promise<NeuralTasteResult> {
  const movieById = new Map(movies.map((movie) => [movie.id, movie]))
  const usableExamples = examples.filter(({ movieId }) => movieById.has(movieId))
  if (usableExamples.length < minimumExamples) {
    return { status: 'insufficient-data', confidence: 0, predictions: {} }
  }

  const tf = await import('@tensorflow/tfjs')
  await tf.setBackend('cpu')
  await tf.ready()
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [inputSize], units: 12, activation: 'relu', kernelInitializer: tf.initializers.glorotUniform({ seed: 17 }) }),
      tf.layers.dense({ units: 6, activation: 'relu', kernelInitializer: tf.initializers.glorotUniform({ seed: 23 }) }),
      tf.layers.dense({ units: 1, activation: 'sigmoid', kernelInitializer: tf.initializers.glorotUniform({ seed: 31 }) }),
    ],
  })
  model.compile({ optimizer: tf.train.adam(0.03), loss: 'meanSquaredError' })

  const inputs = tf.tensor2d(usableExamples.map(({ movieId }) => encodeMovie(movieById.get(movieId)!)))
  const targets = tf.tensor2d(usableExamples.map(({ preference }) => [Math.max(0, Math.min(1, preference))]))

  try {
    const history = await model.fit(inputs, targets, {
      epochs: options.epochs ?? 20,
      batchSize: Math.min(4, usableExamples.length),
      shuffle: true,
      verbose: 0,
    })
    const inferenceInputs = tf.tensor2d(movies.map(encodeMovie))
    const predictionTensor = model.predict(inferenceInputs) as Tensor
    const hiddenOneModel = tf.model({ inputs: model.inputs, outputs: model.layers[0].output })
    const hiddenTwoModel = tf.model({ inputs: model.inputs, outputs: model.layers[1].output })
    const hiddenOneTensor = hiddenOneModel.predict(inferenceInputs) as Tensor
    const hiddenTwoTensor = hiddenTwoModel.predict(inferenceInputs) as Tensor
    const predictionValues = await predictionTensor.data()
    const hiddenOneValues = await hiddenOneTensor.array() as number[][]
    const hiddenTwoValues = await hiddenTwoTensor.array() as number[][]
    predictionTensor.dispose()
    hiddenOneTensor.dispose()
    hiddenTwoTensor.dispose()
    inferenceInputs.dispose()
    const finalLoss = Number(history.history.loss.at(-1) ?? 1)
    const dataConfidence = Math.min(1, usableExamples.length / 12)
    const fitConfidence = Math.max(0, 1 - Math.min(finalLoss, 1))
    const confidence = Math.round(dataConfidence * fitConfidence * 100)
    const predictions = Object.fromEntries(movies.map((movie, index) => [movie.id, Math.round(predictionValues[index] * 100)]))
    const activations = Object.fromEntries(movies.map((movie, index) => [movie.id, { hiddenOne: hiddenOneValues[index], hiddenTwo: hiddenTwoValues[index] }]))
    return { status: 'ready', confidence, predictions, activations, loss: finalLoss, examples: usableExamples.length }
  } catch {
    return { status: 'error', confidence: 0, predictions: {} }
  } finally {
    inputs.dispose()
    targets.dispose()
    model.dispose()
  }
}

const movieKey = (title: string, year?: number) => `${title.trim().toLowerCase()}::${year ?? ''}`

export function buildNeuralExamples(
  movies: Movie[],
  profile: Profile,
  events: FeedbackEvent[],
): NeuralExample[] {
  const movieIdByKey = new Map(movies.map((movie) => [movieKey(movie.title, movie.year), movie.id]))
  const labels = new Map<string, number>()
  profile.ratings.forEach((rating) => {
    const movieId = movieIdByKey.get(movieKey(rating.title, rating.year))
    if (movieId) labels.set(movieId, Math.max(0, Math.min(1, rating.rating / 5)))
  })
  events
    .filter(({ profileId }) => profileId === profile.id)
    .sort((left, right) => left.timestamp - right.timestamp)
    .forEach(({ movieId, rejectedMovieId, action }) => {
      const preference = action === 'skip' ? 0 : action === 'pick' ? 0.75 : action === 'love' ? 1 : 0.8
      labels.set(movieId, preference)
      if (action === 'comparison' && rejectedMovieId) labels.set(rejectedMovieId, 0.2)
    })
  return [...labels].map(([movieId, preference]) => ({ movieId, preference }))
}