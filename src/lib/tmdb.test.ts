import { describe, expect, it } from 'vitest'
import { normalizeTmdbMovie } from './tmdb'

describe('normalizeTmdbMovie', () => {
  it('normalizes metadata and deduplicates Spain streaming providers', () => {
    const movie = normalizeTmdbMovie({
      id: 10,
      title: 'Example',
      release_date: '2022-02-01',
      runtime: 95,
      vote_average: 8.2,
      poster_path: '/poster.jpg',
      genres: [{ name: 'Drama' }],
      original_language: 'es',
      credits: { crew: [{ job: 'Director', name: 'Director Name' }] },
      providers: { ES: { flatrate: [{ provider_name: 'Filmin' }, { provider_name: 'Amazon Prime Video' }], ads: [{ provider_name: 'Filmin' }, { provider_name: 'Amazon Prime Video with Ads' }] } },
    })

    expect(movie).toMatchObject({
      id: 'tmdb-10',
      title: 'Example',
      year: 2022,
      rating: 4.1,
      platforms: ['Filmin', 'Prime Video'],
      director: 'Director Name',
      language: 'Spanish',
    })
  })
})