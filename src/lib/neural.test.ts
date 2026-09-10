import { describe, expect, it } from 'vitest'
import { buildNeuralExamples, encodeMovie, trainNeuralTaste, type NeuralExample } from './neural'
import type { Movie, Profile } from './recommendation'

const movies: Movie[] = [
  { id: 'drama-a', title: 'Drama A', year: 2020, runtime: 110, genres: ['Drama'], platforms: ['Filmin'], rating: 4, director: 'One', language: 'Spanish', moods: ['Reflective'] },
  { id: 'drama-b', title: 'Drama B', year: 2021, runtime: 115, genres: ['Drama'], platforms: ['Filmin'], rating: 3.8, director: 'Two', language: 'Spanish', moods: ['Reflective'] },
  { id: 'comedy-a', title: 'Comedy A', year: 2022, runtime: 90, genres: ['Comedy'], platforms: ['Netflix'], rating: 4, director: 'Three', language: 'English', moods: ['Playful'] },
  { id: 'comedy-b', title: 'Comedy B', year: 2023, runtime: 95, genres: ['Comedy'], platforms: ['Netflix'], rating: 3.9, director: 'Four', language: 'English', moods: ['Playful'] },
]

describe('neural taste model', () => {
  it('encodes arbitrary movie metadata into a stable fixed-size vector', () => {
    const first = encodeMovie(movies[0])

    expect(first).toHaveLength(34)
    expect(encodeMovie(movies[0])).toEqual(first)
    expect(encodeMovie(movies[1])).not.toEqual(first)
  })

  it('reports insufficient data instead of training an unreliable model', async () => {
    const result = await trainNeuralTaste(movies, [{ movieId: 'drama-a', preference: 1 }])

    expect(result).toEqual({ status: 'insufficient-data', confidence: 0, predictions: {} })
  })

  it('builds deduplicated labels from matched ratings and newer feedback', () => {
    const profile: Profile = {
      id: 'maya', name: 'Maya', color: '#fff', watched: [], preferredGenres: [],
      ratings: [{ title: 'Drama A', year: 2020, rating: 4 }],
    }
    const examples = buildNeuralExamples(movies, profile, [
      { id: '1', profileId: 'maya', movieId: 'drama-a', action: 'skip', timestamp: 2 },
      { id: '2', profileId: 'maya', movieId: 'comedy-a', action: 'love', timestamp: 3 },
      { id: '3', profileId: 'leo', movieId: 'comedy-b', action: 'pick', timestamp: 4 },
      { id: '4', profileId: 'maya', movieId: 'comedy-b', rejectedMovieId: 'drama-b', action: 'comparison', timestamp: 5 },
    ])

    expect(examples).toEqual([
      { movieId: 'drama-a', preference: 0 },
      { movieId: 'comedy-a', preference: 1 },
      { movieId: 'comedy-b', preference: 0.8 },
      { movieId: 'drama-b', preference: 0.2 },
    ])
  })

  it('trains a real model and returns bounded predictions', async () => {
    const examples: NeuralExample[] = [
      { movieId: 'drama-a', preference: 0.1 },
      { movieId: 'drama-b', preference: 0.2 },
      { movieId: 'comedy-a', preference: 0.9 },
      { movieId: 'comedy-b', preference: 1 },
    ]

    const result = await trainNeuralTaste(movies, examples, { epochs: 12 })

    expect(result.status).toBe('ready')
    expect(Object.keys(result.predictions)).toHaveLength(4)
    expect(Object.values(result.predictions).every((score) => score >= 0 && score <= 100)).toBe(true)
    expect(result.activations?.['comedy-a'].hiddenOne).toHaveLength(12)
    expect(result.activations?.['comedy-a'].hiddenTwo).toHaveLength(6)
  })
})