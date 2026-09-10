import 'fake-indexeddb/auto'
import { beforeEach, describe, expect, it } from 'vitest'
import { clearAllLocalData, clearSession, decodeSharedRoom, encodeSharedRoom, loadCachedCatalog, loadSession, removeSeedProfiles, saveCachedCatalog, saveSession, type SessionSnapshot } from './session'
import type { Movie } from './recommendation'

const snapshot: SessionSnapshot = {
  profiles: [{
    id: 'maya',
    name: 'Maya',
    color: '#f36b35',
    ratings: [{ title: 'Moonlight', year: 2016, rating: 5 }],
    watched: [{ title: 'Moonlight', year: 2016 }],
    preferredGenres: ['Drama'],
    tasteWeights: { Drama: 0.5 },
  }],
  maxRuntime: 120,
  activeGenres: ['Drama'],
  activePlatforms: ['Filmin'],
  activeMoods: ['Gentle'],
  rankingMode: 'least-misery',
  activeProfileId: 'maya',
  nodePositions: { maya: { x: 10, y: 20 } },
  feedbackEvents: [],
}

describe('session persistence', () => {
  beforeEach(() => clearSession())

  it('round-trips a complete session through IndexedDB', async () => {
    await saveSession(snapshot)

    expect(await loadSession()).toEqual(snapshot)
  })

  it('removes seeded demos when a real imported profile exists', () => {
    const realProfile = { ...snapshot.profiles[0], id: 'moviefan-1', name: 'moviefan' }
    const profiles = [
      { ...snapshot.profiles[0], id: 'maya', name: 'Maya' },
      { ...snapshot.profiles[0], id: 'leo', name: 'Leo' },
      realProfile,
    ]

    expect(removeSeedProfiles(profiles)).toEqual([realProfile])
    expect(removeSeedProfiles(profiles.slice(0, 2))).toHaveLength(2)
  })

  it('shares learned preferences without raw viewing history', () => {
    const encoded = encodeSharedRoom(snapshot)
    const shared = decodeSharedRoom(encoded)

    expect(shared.profiles[0].tasteWeights).toEqual({ Drama: 0.5 })
    expect(shared.profiles[0]).not.toHaveProperty('ratings')
    expect(shared.profiles[0]).not.toHaveProperty('watched')
  })
})

describe('catalog cache', () => {
  const movies: Movie[] = [{ id: 'one', title: 'One', year: 2020, runtime: 90, genres: [], platforms: [], rating: 4 }]

  it('returns a fresh cached catalog', async () => {
    await saveCachedCatalog(movies, 1_000)

    expect(await loadCachedCatalog(1_500, 1_000)).toEqual(movies)
  })

  it('rejects an expired cached catalog', async () => {
    await saveCachedCatalog(movies, 1_000)

    expect(await loadCachedCatalog(3_000, 1_000)).toBeUndefined()
  })

  it('deletes session and cached catalog together', async () => {
    await saveSession(snapshot)
    await saveCachedCatalog(movies)

    await clearAllLocalData()

    expect(await loadSession()).toBeUndefined()
    expect(await loadCachedCatalog()).toBeUndefined()
  })
})