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
      backdrop_path: '/backdrop.jpg',
      overview: 'A richly textured example.',
      genres: [{ name: 'Drama' }],
      original_language: 'es',
      credits: { crew: [{ job: 'Director', name: 'Director Name' }], cast: [{ name: 'Second Actor', order: 2 }, { name: 'Lead Actor', order: 0 }] },
      keywords: { keywords: [{ name: 'complicated women' }, { name: 'beautiful cinematography' }] },
      videos: { results: [{ key: 'trailer-key', site: 'YouTube', type: 'Trailer', official: true }] },
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
      overview: 'A richly textured example.',
      keywords: ['complicated women', 'beautiful cinematography'],
      cast: ['Lead Actor', 'Second Actor'],
      backdrop: 'https://image.tmdb.org/t/p/w1280/backdrop.jpg',
      trailerUrl: 'https://www.youtube.com/watch?v=trailer-key',
    })
  })
})