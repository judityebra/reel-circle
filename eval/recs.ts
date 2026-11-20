/**
 * Offline recommendation benchmark on a seeded synthetic world.
 *
 *   npm run eval:recs                 # default configuration
 *   npm run eval:recs -- --users 20   # quicker run
 *
 * Flags: --seed, --users, --movies, --directors, --held-out, --triples, --group-labels, --json <path>
 */
import { writeFileSync } from 'node:fs'
import { trainGraphNeuralTaste } from '../src/lib/graph-neural'
import { buildNeuralExamples, trainNeuralTaste } from '../src/lib/neural'
import {
  applyTasteFeedback,
  scoreCandidates,
  type Movie,
  type Profile,
  type RankingMode,
  type TasteFeedback,
} from '../src/lib/recommendation'
import { bootstrapCI, formatInterval, mean, ndcgAtK, precisionAtK, spearman, type Interval } from './metrics'
import { createRng, patchMathRandom, shuffle } from './rng'
import { generateWorld, type SyntheticMovie, type SyntheticUser } from './synthetic'

const LABEL_COUNTS = [8, 16, 32] as const
const K = 10
const LIKED = 4
const MISERY = 2.5
/** Fallback genres from App.tsx when no rated film matches the catalog. */
const APP_DEFAULT_GENRES = ['Drama', 'Comedy', 'Mystery', 'Romance', 'Sci-Fi']

const METHODS = ['random', 'popularity', 'scorer', 'scorer+feedback', 'mlp', 'gcn', 'mlp-hybrid', 'gcn-hybrid'] as const
type Method = (typeof METHODS)[number]

interface Config {
  seed: number
  users: number
  movies: number
  directors: number
  heldOut: number
  triples: number
  groupLabels: number
  json?: string
}

function parseArgs(argv: string[]): Config {
  const config: Config = { seed: 2026, users: 60, movies: 150, directors: 30, heldOut: 50, triples: 200, groupLabels: 16 }
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index].replace(/^--/, '')
    const value = argv[index + 1]
    if (flag === 'json') { config.json = value; index += 1; continue }
    const key = flag.replace(/-(\w)/g, (_, letter: string) => letter.toUpperCase()) as keyof Config
    if (key in config && value !== undefined) {
      ;(config as unknown as Record<string, number>)[key] = Number(value)
      index += 1
    }
  }
  return config
}

/** The feedback mapping used to replay hidden ratings through the app's skip/pick/love controls. */
export function ratingToFeedback(rating: number): TasteFeedback {
  if (rating >= 4.5) return 'love'
  if (rating >= 3) return 'pick'
  return 'skip'
}

/** preferredGenres exactly as App.tsx derives them on Letterboxd import (catalog order, rating ignored). */
function derivePreferredGenres(catalog: Movie[], labelled: Movie[]): string[] {
  const titles = new Set(labelled.map(({ title }) => title.toLowerCase()))
  const genres = [...new Set(catalog.filter((movie) => titles.has(movie.title.toLowerCase())).flatMap((movie) => movie.genres))].slice(0, 3)
  return genres.length ? genres : APP_DEFAULT_GENRES.slice(0, 2)
}

function shippedProfile(user: SyntheticUser, catalog: Movie[], labelled: SyntheticMovie[]): Profile {
  return {
    id: user.id,
    name: user.id,
    color: '#000000',
    ratings: labelled.map(({ title, year, id }) => ({ title, year, rating: user.ratings[id] })),
    watched: labelled.map(({ title, year }) => ({ title, year })),
    preferredGenres: derivePreferredGenres(catalog, labelled),
  }
}

function feedbackProfile(user: SyntheticUser, catalog: Movie[], labelled: SyntheticMovie[]): Profile {
  return labelled.reduce(
    (profile, movie) => applyTasteFeedback(profile, movie, ratingToFeedback(user.ratings[movie.id])),
    shippedProfile(user, catalog, labelled),
  )
}

const scorerScores = (heldOut: Movie[], profile: Profile, options: Parameters<typeof scoreCandidates>[2] = {}) =>
  Object.fromEntries(scoreCandidates(heldOut, [profile], options).map(({ movie, score }) => [movie.id, score]))

interface UserMetrics { ndcg: number; precision: number; spearman: number }

/** Rank held-out movies by score (stable, so ties keep the randomised held-out order). */
function evaluateRanking(heldOut: Movie[], scores: Record<string, number>, user: SyntheticUser): UserMetrics {
  const ranked = heldOut
    .map((movie, index) => ({ movie, index, score: scores[movie.id] ?? 0 }))
    .sort((left, right) => right.score - left.score || left.index - right.index)
  const truth = ranked.map(({ movie }) => user.ratings[movie.id])
  return {
    ndcg: ndcgAtK(truth, truth, K),
    precision: precisionAtK(truth, K, LIKED),
    spearman: spearman(ranked.map(({ score }) => score), truth),
  }
}

async function silently<T>(task: () => Promise<T>): Promise<T> {
  // TensorFlow.js prints a one-off "use tfjs-node" banner; keep benchmark output clean.
  const warn = console.warn
  console.warn = () => {}
  try {
    return await task()
  } finally {
    console.warn = warn
  }
}

async function runIndividualBenchmark(world: ReturnType<typeof generateWorld>, heldOut: SyntheticMovie[], labelOrder: Map<string, SyntheticMovie[]>, rng: () => number) {
  const catalog = world.movies
  const results: Record<number, Record<Method, UserMetrics[]>> = {}
  const confidence: Record<number, { mlp: number[]; gcn: number[]; notReady: number }> = {}

  for (const labels of LABEL_COUNTS) {
    results[labels] = Object.fromEntries(METHODS.map((method) => [method, []])) as unknown as Record<Method, UserMetrics[]>
    confidence[labels] = { mlp: [], gcn: [], notReady: 0 }
    const started = Date.now()
    for (const user of world.users) {
      const labelled = labelOrder.get(user.id)!.slice(0, labels)
      const shipped = shippedProfile(user, catalog, labelled)
      const examples = buildNeuralExamples(catalog, shipped, [])
      const mlp = await silently(() => trainNeuralTaste(catalog, examples))
      const gcn = await silently(() => trainGraphNeuralTaste(catalog, examples))
      if (mlp.status !== 'ready') confidence[labels].notReady += 1
      if (gcn.status !== 'ready') confidence[labels].notReady += 1
      confidence[labels].mlp.push(mlp.confidence)
      confidence[labels].gcn.push(gcn.confidence)

      const scoresByMethod: Record<Method, Record<string, number>> = {
        random: Object.fromEntries(heldOut.map(({ id }) => [id, rng()])),
        popularity: Object.fromEntries(heldOut.map(({ id, rating }) => [id, rating])),
        scorer: scorerScores(heldOut, shipped),
        'scorer+feedback': scorerScores(heldOut, feedbackProfile(user, catalog, labelled)),
        mlp: mlp.predictions,
        gcn: gcn.predictions,
        'mlp-hybrid': scorerScores(heldOut, shipped, { neuralScores: { [user.id]: mlp.predictions }, neuralBlend: mlp.confidence / 200 }),
        'gcn-hybrid': scorerScores(heldOut, shipped, { neuralScores: { [user.id]: gcn.predictions }, neuralBlend: gcn.confidence / 200 }),
      }
      METHODS.forEach((method) => results[labels][method].push(evaluateRanking(heldOut, scoresByMethod[method], user)))
    }
    process.stderr.write(`  ${labels} labels/user: ${world.users.length} users in ${((Date.now() - started) / 1000).toFixed(1)}s\n`)
  }
  return { results, confidence }
}

const GROUP_STRATEGIES: RankingMode[] = ['balanced', 'average', 'least-misery', 'nash']

interface GroupOutcome { mean: number; min: number; misery: number }

function outcomeFor(movie: Movie, members: SyntheticUser[]): GroupOutcome {
  const ratings = members.map((member) => member.ratings[movie.id])
  return { mean: mean(ratings), min: Math.min(...ratings), misery: ratings.some((rating) => rating <= MISERY) ? 1 : 0 }
}

function runGroupBenchmark(config: Config, world: ReturnType<typeof generateWorld>, heldOut: SyntheticMovie[], labelOrder: Map<string, SyntheticMovie[]>, rng: () => number) {
  const profiles = new Map(world.users.map((user) => [user.id, feedbackProfile(user, world.movies, labelOrder.get(user.id)!.slice(0, config.groupLabels))]))
  const rows: Record<string, GroupOutcome[]> = {
    random: [],
    ...Object.fromEntries(GROUP_STRATEGIES.map((strategy) => [strategy, []])),
    'oracle (max mean)': [],
    'oracle (max min)': [],
  }
  for (let triple = 0; triple < config.triples; triple += 1) {
    const members = shuffle(rng, world.users).slice(0, 3)
    const memberProfiles = members.map(({ id }) => profiles.get(id)!)
    GROUP_STRATEGIES.forEach((strategy) => {
      const top = scoreCandidates(heldOut, memberProfiles, { rankingMode: strategy })[0]
      rows[strategy].push(outcomeFor(top.movie, members))
    })
    rows.random.push(outcomeFor(heldOut[Math.floor(rng() * heldOut.length)], members))
    const outcomes = heldOut.map((movie) => outcomeFor(movie, members))
    rows['oracle (max mean)'].push(outcomes.reduce((best, next) => next.mean > best.mean ? next : best))
    rows['oracle (max min)'].push(outcomes.reduce((best, next) => next.min > best.min || (next.min === best.min && next.mean > best.mean) ? next : best))
  }
  return rows
}

async function main() {
  const config = parseArgs(process.argv.slice(2))
  const restore = patchMathRandom(config.seed ^ 0x9e3779b9)
  const rng = createRng(config.seed)
  const world = generateWorld(rng, config)
  const shuffled = shuffle(rng, world.movies)
  const heldOut = shuffled.slice(0, config.heldOut)
  const trainPool = shuffled.slice(config.heldOut)
  // Nested label sets: the 8 labels are a prefix of the 16, which are a prefix of the 32.
  const labelOrder = new Map(world.users.map((user) => [user.id, shuffle(rng, trainPool)]))
  const ciRng = createRng(config.seed + 1)

  process.stderr.write(`Reel Circle recommendation benchmark (seed ${config.seed})\n`)
  const { results, confidence } = await runIndividualBenchmark(world, heldOut, labelOrder, rng)
  const groups = runGroupBenchmark(config, world, heldOut, labelOrder, rng)
  restore()

  const ci = (values: number[]) => bootstrapCI(values, ciRng)
  const lines: string[] = []
  const heldOutRatings = world.users.flatMap((user) => heldOut.map(({ id }) => user.ratings[id]))
  lines.push(`# Recommendation benchmark`, '')
  lines.push(`seed=${config.seed} · ${config.movies} movies · ${config.directors} directors · ${config.users} users · ${config.heldOut} held-out movies (fixed) · ${config.movies - config.heldOut} label pool`)
  lines.push(`held-out hidden ratings: mean ${mean(heldOutRatings).toFixed(2)}, ${(heldOutRatings.filter((rating) => rating >= LIKED).length / heldOutRatings.length * 100).toFixed(1)}% liked (>= ${LIKED})`)
  lines.push(`NDCG@${K} (linear gain = hidden rating), Precision@${K} (liked >= ${LIKED}), Spearman over all held-out movies. Mean [95% bootstrap CI] across users.`, '')

  const json: Record<string, unknown> = { config, individual: {}, groups: {} }
  for (const labels of LABEL_COUNTS) {
    lines.push(`## ${labels} labels per user`, '')
    lines.push(`MLP confidence ${mean(confidence[labels].mlp).toFixed(1)} (blend ${(mean(confidence[labels].mlp) / 2).toFixed(1)}%) · GCN confidence ${mean(confidence[labels].gcn).toFixed(1)} (blend ${(mean(confidence[labels].gcn) / 2).toFixed(1)}%) · untrained models: ${confidence[labels].notReady}`, '')
    lines.push(`| Method | NDCG@${K} | Precision@${K} | Spearman |`, '| --- | --- | --- | --- |')
    const table: Record<string, Record<string, Interval>> = {}
    METHODS.forEach((method) => {
      const rows = results[labels][method]
      const summary = {
        ndcg: ci(rows.map(({ ndcg }) => ndcg)),
        precision: ci(rows.map(({ precision }) => precision)),
        spearman: ci(rows.map(({ spearman: value }) => value)),
      }
      table[method] = summary
      lines.push(`| ${method} | ${formatInterval(summary.ndcg)} | ${formatInterval(summary.precision)} | ${formatInterval(summary.spearman)} |`)
    })
    ;(json.individual as Record<number, unknown>)[labels] = { confidence: { mlp: mean(confidence[labels].mlp), gcn: mean(confidence[labels].gcn) }, metrics: table }
    lines.push('')
  }

  lines.push(`## Group fairness (${config.triples} random triples, members use scorer+feedback profiles from ${config.groupLabels} labels)`, '')
  lines.push('Top pick scored on members\' hidden ratings. Mean [95% bootstrap CI] across triples.', '')
  lines.push(`| Strategy | Mean member rating | Min member rating | Someone rates <= ${MISERY} |`, '| --- | --- | --- | --- |')
  Object.entries(groups).forEach(([strategy, rows]) => {
    const summary = { mean: ci(rows.map((row) => row.mean)), min: ci(rows.map((row) => row.min)), misery: ci(rows.map((row) => row.misery * 100)) }
    ;(json.groups as Record<string, unknown>)[strategy] = summary
    lines.push(`| ${strategy} | ${formatInterval(summary.mean, 2)} | ${formatInterval(summary.min, 2)} | ${formatInterval(summary.misery, 1)}% |`)
  })

  console.log(lines.join('\n'))
  if (config.json) writeFileSync(config.json, `${JSON.stringify(json, null, 2)}\n`)
}

await main()
