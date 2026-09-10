import { describe, expect, it } from 'vitest'
import { buildMovieFeatureGraph, trainGraphNeuralTaste } from './graph-neural'
import type { NeuralExample } from './neural'
import type { Movie } from './recommendation'

const movies: Movie[] = [
  { id: 'drama-a', title: 'Drama A', year: 2020, runtime: 110, genres: ['Drama'], platforms: ['Filmin'], rating: 4, director: 'One', language: 'Spanish', moods: ['Reflective'] },
  { id: 'drama-b', title: 'Drama B', year: 2021, runtime: 115, genres: ['Drama'], platforms: ['Filmin'], rating: 3.8, director: 'Two', language: 'Spanish', moods: ['Reflective'] },
  { id: 'comedy-a', title: 'Comedy A', year: 2022, runtime: 90, genres: ['Comedy'], platforms: ['Netflix'], rating: 4, director: 'Three', language: 'English', moods: ['Playful'] },
  { id: 'comedy-b', title: 'Comedy B', year: 2023, runtime: 95, genres: ['Comedy'], platforms: ['Netflix'], rating: 3.9, director: 'Four', language: 'English', moods: ['Playful'] },
]

describe('graph neural recommender', () => {
  it('builds a symmetric normalized movie-feature graph with self loops', () => {
    const graph = buildMovieFeatureGraph(movies)
    const dramaIndex = graph.nodeIds.indexOf('feature:genre:Drama')
    const movieIndex = graph.nodeIds.indexOf('movie:drama-a')

    expect(graph.movieIndices).toHaveLength(4)
    expect(dramaIndex).toBeGreaterThan(-1)
    expect(graph.adjacency[movieIndex][dramaIndex]).toBeGreaterThan(0)
    expect(graph.adjacency[dramaIndex][movieIndex]).toBe(graph.adjacency[movieIndex][dramaIndex])
    expect(graph.adjacency[movieIndex][movieIndex]).toBeGreaterThan(0)
  })

  it('refuses to train without enough labeled movie nodes', async () => {
    const result = await trainGraphNeuralTaste(movies, [{ movieId: 'drama-a', preference: 1 }])

    expect(result).toEqual({ status: 'insufficient-data', confidence: 0, predictions: {}, embeddings: {} })
  })

  it('trains a two-layer GCN and exports movie embeddings', async () => {
    const examples: NeuralExample[] = [
      { movieId: 'drama-a', preference: 0.1 },
      { movieId: 'drama-b', preference: 0.2 },
      { movieId: 'comedy-a', preference: 0.9 },
      { movieId: 'comedy-b', preference: 1 },
    ]

    const result = await trainGraphNeuralTaste(movies, examples, { epochs: 16 })

    expect(result.status).toBe('ready')
    expect(Object.values(result.predictions).every((score) => score >= 0 && score <= 100)).toBe(true)
    expect(result.embeddings['comedy-a']).toHaveLength(12)
    expect(result.graphStats).toMatchObject({ movieNodes: 4, layers: 2 })
  })
})