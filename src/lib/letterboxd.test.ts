import { describe, expect, it } from 'vitest'
import JSZip from 'jszip'
import { parseLetterboxdCsv, parseLetterboxdExport, parseLetterboxdRss, parseLetterboxdZip } from './letterboxd'

describe('parseLetterboxdCsv', () => {
  it('parses ratings and watched films from a Letterboxd export', () => {
    const csv = `Date,Name,Year,Letterboxd URI,Rating\n2026-01-01,"Paris, Texas",1984,https://boxd.it/29Ts,4.5\n2026-01-02,Clue,1985,https://boxd.it/1Sao,`

    expect(parseLetterboxdCsv(csv)).toEqual({
      ratings: [{ title: 'Paris, Texas', year: 1984, rating: 4.5 }],
      watched: [
        { title: 'Paris, Texas', year: 1984 },
        { title: 'Clue', year: 1985 },
      ],
    })
  })

  it('ignores malformed rows instead of importing bad data', () => {
    const csv = `Name,Year,Rating\n,2020,5\nArrival,nope,4\nAftersun,2022,excellent`

    expect(parseLetterboxdCsv(csv)).toEqual({ ratings: [], watched: [] })
  })
})

describe('parseLetterboxdRss', () => {
  const rss = `<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0" xmlns:letterboxd="https://letterboxd.com" xmlns:tmdb="https://themoviedb.org">
      <channel>
        <title>Letterboxd - moviefan</title>
        <link>https://letterboxd.com/moviefan/</link>
        <item><letterboxd:filmTitle>Sinners</letterboxd:filmTitle><letterboxd:filmYear>2025</letterboxd:filmYear><letterboxd:memberRating>3.5</letterboxd:memberRating><tmdb:movieId>1233413</tmdb:movieId></item>
        <item><letterboxd:filmTitle>Love Actually</letterboxd:filmTitle><letterboxd:filmYear>2003</letterboxd:filmYear><letterboxd:memberRating>4.5</letterboxd:memberRating><tmdb:movieId>508</tmdb:movieId></item>
        <item><letterboxd:filmTitle>Love Actually</letterboxd:filmTitle><letterboxd:filmYear>2003</letterboxd:filmYear><letterboxd:memberRating>4</letterboxd:memberRating><tmdb:movieId>508</tmdb:movieId></item>
      </channel>
    </rss>`

  it('parses and deduplicates ratings and watched films from RSS activity', () => {
    expect(parseLetterboxdRss(rss)).toEqual({
      profileName: 'moviefan',
      ratings: [
        { title: 'Sinners', year: 2025, rating: 3.5, tmdbId: 1233413 },
        { title: 'Love Actually', year: 2003, rating: 4.5, tmdbId: 508 },
      ],
      watched: [
        { title: 'Sinners', year: 2025, tmdbId: 1233413 },
        { title: 'Love Actually', year: 2003, tmdbId: 508 },
      ],
    })
  })

  it('auto-detects RSS and CSV exports from content', () => {
    expect(parseLetterboxdExport(rss)).toEqual(parseLetterboxdRss(rss))
    expect(parseLetterboxdExport('Name,Year,Rating\nClue,1985,4')).toEqual(parseLetterboxdCsv('Name,Year,Rating\nClue,1985,4'))
  })

  it('rejects unsupported content', () => {
    expect(() => parseLetterboxdExport('not an export')).toThrow('Unsupported Letterboxd export format')
  })
})

describe('parseLetterboxdZip', () => {
  it('combines complete ratings, watched, diary, and profile data', async () => {
    const zip = new JSZip()
    zip.file('profile.csv', 'Date Joined,Username,Given Name\n2020-01-01,moviefan,Alex')
    zip.file('watched.csv', 'Date,Name,Year,Letterboxd URI\n2020-01-01,Clue,1985,https://boxd.it/1Sao\n2020-01-02,Arrival,2016,https://boxd.it/abc')
    zip.file('ratings.csv', 'Date,Name,Year,Letterboxd URI,Rating\n2020-01-03,Clue,1985,https://boxd.it/1Sao,4.5')
    zip.file('diary.csv', 'Date,Name,Year,Letterboxd URI,Rating,Rewatch\n2020-01-04,Clue,1985,https://boxd.it/1Sao,4,Yes\n2020-01-05,Aftersun,2022,https://boxd.it/x,5,No')
    const archive = await zip.generateAsync({ type: 'uint8array' })

    expect(await parseLetterboxdZip(archive)).toEqual({
      profileName: 'moviefan',
      ratings: [{ title: 'Clue', year: 1985, rating: 4.5 }],
      watched: [
        { title: 'Clue', year: 1985 },
        { title: 'Arrival', year: 2016 },
        { title: 'Aftersun', year: 2022 },
      ],
    })
  })

  it('rejects archives without Letterboxd history files', async () => {
    const zip = new JSZip()
    zip.file('readme.txt', 'nothing useful')
    const archive = await zip.generateAsync({ type: 'uint8array' })

    await expect(parseLetterboxdZip(archive)).rejects.toThrow('No Letterboxd history found')
  })
})