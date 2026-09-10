import Papa from 'papaparse'
import { XMLParser } from 'fast-xml-parser'
import type { FilmRef, Rating } from './recommendation'

interface LetterboxdRow {
  Name?: string
  Year?: string
  Rating?: string
}

export interface LetterboxdImport {
  ratings: Rating[]
  watched: FilmRef[]
  profileName?: string
}

interface LetterboxdRssItem {
  filmTitle?: string
  filmYear?: number | string
  memberRating?: number | string
  movieId?: number | string
}

interface LetterboxdRss {
  rss?: {
    channel?: {
      title?: string
      link?: unknown
      item?: LetterboxdRssItem | LetterboxdRssItem[]
    }
  }
}

export function parseLetterboxdCsv(csv: string): LetterboxdImport {
  const parsed = Papa.parse<LetterboxdRow>(csv, {
    header: true,
    skipEmptyLines: true,
    transformHeader: (header) => header.trim(),
  })

  return parsed.data.reduce<LetterboxdImport>(
    (result, row) => {
      const title = row.Name?.trim()
      const year = Number(row.Year)
      const ratingText = row.Rating?.trim() ?? ''
      const rating = Number(ratingText)

      if (!title || !Number.isInteger(year) || year < 1888 || (ratingText && !Number.isFinite(rating))) {
        return result
      }

      result.watched.push({ title, year })
      if (ratingText && rating >= 0.5 && rating <= 5) {
        result.ratings.push({ title, year, rating })
      }
      return result
    },
    { ratings: [], watched: [] },
  )
}

export function parseLetterboxdRss(xml: string): LetterboxdImport {
  const parser = new XMLParser({
    ignoreAttributes: false,
    removeNSPrefix: true,
    parseTagValue: true,
    trimValues: true,
  })
  const parsed = parser.parse(xml) as LetterboxdRss
  const channel = parsed.rss?.channel
  if (!channel) throw new Error('Invalid Letterboxd RSS feed')
  const items = channel.item ? (Array.isArray(channel.item) ? channel.item : [channel.item]) : []
  const watchedByKey = new Map<string, FilmRef>()
  const ratingsByKey = new Map<string, Rating>()

  items.forEach((item) => {
    const title = String(item.filmTitle ?? '').trim()
    const year = Number(item.filmYear)
    const rating = Number(item.memberRating)
    const tmdbId = Number(item.movieId)
    if (!title || !Number.isInteger(year) || year < 1888) return
    const key = `${title.toLocaleLowerCase()}::${year}`
    const film = { title, year, ...(Number.isInteger(tmdbId) && tmdbId > 0 ? { tmdbId } : {}) }
    if (!watchedByKey.has(key)) watchedByKey.set(key, film)
    if (!ratingsByKey.has(key) && Number.isFinite(rating) && rating >= 0.5 && rating <= 5) {
      ratingsByKey.set(key, { ...film, rating })
    }
  })

  const links = Array.isArray(channel.link) ? channel.link : [channel.link]
  const profileLink = links.find((link): link is string => typeof link === 'string' && /letterboxd\.com\/[\w-]+\/?$/i.test(link))
  const username = profileLink?.match(/letterboxd\.com\/([\w-]+)/i)?.[1]
  const channelTitle = String(channel.title ?? '')
  const profileName = username ?? (channelTitle.replace(/^Letterboxd\s*-\s*/i, '').trim() || undefined)
  return {
    ratings: [...ratingsByKey.values()],
    watched: [...watchedByKey.values()],
    ...(profileName ? { profileName } : {}),
  }
}

export function parseLetterboxdExport(content: string): LetterboxdImport {
  const trimmed = content.trimStart()
  if (trimmed.startsWith('<?xml') || /^<rss[\s>]/i.test(trimmed)) {
    return parseLetterboxdRss(content)
  }
  const header = trimmed.split(/\r?\n/, 1)[0] ?? ''
  if (/(^|,)Name(,|$)/i.test(header) && /(^|,)Year(,|$)/i.test(header)) {
    return parseLetterboxdCsv(content)
  }
  throw new Error('Unsupported Letterboxd export format')
}

const filmKey = ({ title, year }: FilmRef) => `${title.trim().toLocaleLowerCase()}::${year ?? ''}`

export async function parseLetterboxdZip(data: ArrayBuffer | Uint8Array): Promise<LetterboxdImport> {
  const { default: JSZip } = await import('jszip')
  const zip = await JSZip.loadAsync(data)
  const readEntry = async (name: string) => zip.file(name)?.async('string')
  const [profileCsv, watchedCsv, ratingsCsv, diaryCsv] = await Promise.all([
    readEntry('profile.csv'),
    readEntry('watched.csv'),
    readEntry('ratings.csv'),
    readEntry('diary.csv'),
  ])
  if (!watchedCsv && !ratingsCsv && !diaryCsv) {
    throw new Error('No Letterboxd history found in this archive')
  }

  const watched = new Map<string, FilmRef>()
  const mergeWatched = (content?: string) => {
    if (!content) return
    parseLetterboxdCsv(content).watched.forEach((film) => watched.set(filmKey(film), film))
  }
  mergeWatched(watchedCsv)
  mergeWatched(ratingsCsv)
  mergeWatched(diaryCsv)

  const ratings = ratingsCsv ? parseLetterboxdCsv(ratingsCsv).ratings : []
  const profile = profileCsv
    ? Papa.parse<{ Username?: string }>(profileCsv, { header: true, skipEmptyLines: true }).data[0]
    : undefined
  const profileName = profile?.Username?.trim()

  return {
    ratings,
    watched: [...watched.values()],
    ...(profileName ? { profileName } : {}),
  }
}