import { openDB, type DBSchema } from 'idb'
import type { AgentTraceEntry } from './decision-agent'
import type { ExperimentRecord } from './experiments'
import type { GraphNeuralResult } from './graph-neural'
import type { NeuralTasteResult } from './neural'
import type { Movie, Profile, RankingMode, TasteFeedback } from './recommendation'

export interface FeedbackEvent {
  id: string
  profileId: string
  movieId: string
  rejectedMovieId?: string
  action: TasteFeedback | 'comparison'
  timestamp: number
}

export interface SessionSnapshot {
  profiles: Profile[]
  maxRuntime?: number
  activeGenres: string[]
  activePlatforms: string[]
  activeMoods: string[]
  rankingMode: RankingMode
  activeProfileId: string
  nodePositions: Record<string, { x: number; y: number }>
  feedbackEvents: FeedbackEvent[]
  neuralResults?: Record<string, NeuralTasteResult>
  graphNeuralResults?: Record<string, GraphNeuralResult>
  experiments?: ExperimentRecord[]
  agentTrace?: AgentTraceEntry[]
}

type SharedProfile = Pick<Profile, 'id' | 'name' | 'color' | 'preferredGenres' | 'tasteWeights' | 'featureWeights'>

export interface SharedRoom {
  version: 1
  profiles: SharedProfile[]
  maxRuntime?: number
  activeGenres: string[]
  activePlatforms: string[]
  activeMoods: string[]
  rankingMode: RankingMode
}

interface ReelCircleDatabase extends DBSchema {
  session: {
    key: 'current'
    value: SessionSnapshot
  }
  catalog: {
    key: 'live-es'
    value: { movies: Movie[]; savedAt: number }
  }
}

const database = openDB<ReelCircleDatabase>('reel-circle', 2, {
  upgrade(db) {
    if (!db.objectStoreNames.contains('session')) db.createObjectStore('session')
    if (!db.objectStoreNames.contains('catalog')) db.createObjectStore('catalog')
  },
})

export async function saveSession(snapshot: SessionSnapshot): Promise<void> {
  const db = await database
  await db.put('session', snapshot, 'current')
}

export async function loadSession(): Promise<SessionSnapshot | undefined> {
  const db = await database
  return db.get('session', 'current')
}

export async function clearSession(): Promise<void> {
  const db = await database
  await db.clear('session')
}

const toBase64Url = (value: string) => {
  const bytes = new TextEncoder().encode(value)
  const binary = Array.from(bytes, (byte) => String.fromCharCode(byte)).join('')
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '')
}

const fromBase64Url = (value: string) => {
  const padded = value.replaceAll('-', '+').replaceAll('_', '/').padEnd(Math.ceil(value.length / 4) * 4, '=')
  const bytes = Uint8Array.from(atob(padded), (character) => character.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

export function encodeSharedRoom(snapshot: SessionSnapshot): string {
  const room: SharedRoom = {
    version: 1,
    profiles: snapshot.profiles.map(({ id, name, color, preferredGenres, tasteWeights, featureWeights }) => ({
      id,
      name,
      color,
      preferredGenres,
      tasteWeights,
      featureWeights,
    })),
    maxRuntime: snapshot.maxRuntime,
    activeGenres: snapshot.activeGenres,
    activePlatforms: snapshot.activePlatforms,
    activeMoods: snapshot.activeMoods,
    rankingMode: snapshot.rankingMode,
  }
  return toBase64Url(JSON.stringify(room))
}

export function decodeSharedRoom(encoded: string): SharedRoom {
  const parsed: unknown = JSON.parse(fromBase64Url(encoded))
  if (!parsed || typeof parsed !== 'object' || !('version' in parsed) || parsed.version !== 1 || !('profiles' in parsed) || !Array.isArray(parsed.profiles)) {
    throw new Error('Invalid Reel Circle room')
  }
  return parsed as SharedRoom
}

const seedProfileIds = new Set(['maya', 'leo'])

export function removeSeedProfiles(profiles: Profile[]): Profile[] {
  const hasImportedProfile = profiles.some(({ id }) => !seedProfileIds.has(id))
  return hasImportedProfile ? profiles.filter(({ id }) => !seedProfileIds.has(id)) : profiles
}

export async function saveCachedCatalog(movies: Movie[], savedAt = Date.now()): Promise<void> {
  const db = await database
  await db.put('catalog', { movies, savedAt }, 'live-es')
}

export async function loadCachedCatalog(
  now = Date.now(),
  maxAgeMs = 24 * 60 * 60 * 1_000,
): Promise<Movie[] | undefined> {
  const db = await database
  const cached = await db.get('catalog', 'live-es')
  return cached && now - cached.savedAt <= maxAgeMs ? cached.movies : undefined
}

export async function clearAllLocalData(): Promise<void> {
  const db = await database
  const transaction = db.transaction(['session', 'catalog'], 'readwrite')
  await Promise.all([
    transaction.objectStore('session').clear(),
    transaction.objectStore('catalog').clear(),
    transaction.done,
  ])
}