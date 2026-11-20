import { describe, expect, it } from 'vitest'
import { createRng } from './rng'
import { generateWorld } from './synthetic'

describe('generateWorld', () => {
  const options = { movies: 40, users: 5, directors: 8 }

  it('is deterministic for a seed', () => {
    expect(generateWorld(createRng(3), options)).toEqual(generateWorld(createRng(3), options))
  })

  it('gives every user a half-star hidden rating for every movie', () => {
    const world = generateWorld(createRng(4), options)
    world.users.forEach((user) => {
      expect(Object.keys(user.ratings)).toHaveLength(options.movies)
      Object.values(user.ratings).forEach((rating) => {
        expect(rating).toBeGreaterThanOrEqual(0.5)
        expect(rating).toBeLessThanOrEqual(5)
        expect(rating * 2).toBe(Math.round(rating * 2))
      })
    })
  })

  it('mostly assigns movies to their director home genre', () => {
    const world = generateWorld(createRng(5), { ...options, movies: 200 })
    const homeGenre = new Map(world.directors.map(({ name, homeGenre: genre }) => [name, genre]))
    const matching = world.movies.filter((movie) => movie.genres.includes(homeGenre.get(movie.director!)!)).length
    expect(matching / world.movies.length).toBeGreaterThan(0.75)
  })
})
