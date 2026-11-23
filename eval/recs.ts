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
  type RecommendationOptions,
  type TasteFeedback,
} from '../src/lib/recommendation'
import { bootstrapCI, formatInterval, mean, ndcgAtK, precisionAtK, spearman, type Interval } from './metrics'
import { createRng, patchMathRandom, shuffle } from './rng'
import { generateWorld, type SyntheticMovie, type SyntheticUser, type SyntheticWorld } from './synthetic'

const LABEL_COUNTS = [8, 16, 32] as const
type LabelCount = (typeof LABEL_COUNTS)[number]
const K = 10
const LIKED = 4
const MISERY = 2.5
/** Fallback genres from App.tsx when no rated film matches the catalog. */
const APP_DEFAULT_GENRES = ['Drama', 'Comedy', 'Mystery', 'Romance', 'Sci-Fi']

const METHODS = ['random', 'popularity', 'scorer', 'scorer+feedback', 'mlp', 'gcn', 'mlp-hybrid', 'gcn-hybrid'] as const
type Method = (typeof METHODS)[number]

const GROUP_STRATEGIES: RankingMode[] = ['balanced', 'average', 'least-misery', 'nash']
const GROUP_SIZE = 3

type Scores = Record<string, number>

// ---------------------------------------------------------------------------
// Configuration

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

type NumericFlag = Exclude<keyof Config, 'json'>

const DEFAULT_CONFIG: Required<Omit<Config, 'json'>> = {
  seed: 2026,
  users: 60,
  movies: 150,
  directors: 30,
  heldOut: 50,
  triples: 200,
  groupLabels: 16,
}

const kebabToCamel = (flag: string) => flag.replace(/-(\w)/g, (_, letter: string) => letter.toUpperCase())

function parseArgs(argv: string[]): Config {
  const config: Config = { ...DEFAULT_CONFIG }
  for (let index = 0; index < argv.length; index += 1) {
    const key = kebabToCamel(argv[index].replace(/^--/, ''))
    const value = argv[index + 1]
    if (value === undefined) continue
    if (key === 'json') {
      config.json = value
      index += 1
    } else if (key in DEFAULT_CONFIG) {
      config[key as NumericFlag] = Number(value)
      index += 1
    }
  }
  return config
}

// ---------------------------------------------------------------------------
// Profiles, built the same way the app builds them

/** The feedback mapping used to replay hidden ratings through the app's skip/pick/love controls. */
function ratingToFeedback(rating: number): TasteFeedback {
  if (rating >= 4.5) return 'love'
  if (rating >= 3) return 'pick'
  return 'skip'
}

/** Same confidence (0–100) → blend weight (0–0.5) mapping App.tsx uses for neural hybrids. */
const confidenceToBlend = (confidence: number) => confidence / 200

/** preferredGenres exactly as App.tsx derives them on Letterboxd import (catalog order, rating ignored). */
function derivePreferredGenres(catalog: Movie[], labelled: Movie[]): string[] {
  const titles = new Set(labelled.map(({ title }) => title.toLowerCase()))
  const matched = catalog.filter((movie) => titles.has(movie.title.toLowerCase()))
  const genres = [...new Set(matched.flatMap((movie) => movie.genres))].slice(0, 3)
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

/** A fixed, user-specific label order; taking prefixes keeps the 8/16/32 label sets nested. */
function createLabelPool(order: Map<string, SyntheticMovie[]>) {
  return { take: (user: SyntheticUser, count: number) => order.get(user.id)!.slice(0, count) }
}
type LabelPool = ReturnType<typeof createLabelPool>

// ---------------------------------------------------------------------------
// Individual benchmark

interface UserMetrics { ndcg: number; precision: number; spearman: number }

interface IndividualResult {
  metrics: Record<Method, UserMetrics[]>
  confidence: { mlp: number[]; gcn: number[] }
  notReady: number
}

const scorerScores = (heldOut: Movie[], profile: Profile, options: RecommendationOptions = {}): Scores =>
  Object.fromEntries(scoreCandidates(heldOut, [profile], options).map(({ movie, score }) => [movie.id, score]))

const hybridScores = (heldOut: Movie[], profile: Profile, model: { predictions: Scores; confidence: number }): Scores =>
  scorerScores(heldOut, profile, { neuralScores: { [profile.id]: model.predictions }, neuralBlend: confidenceToBlend(model.confidence) })

/** Rank held-out movies by score (stable, so ties keep the randomised held-out order). */
function evaluateRanking(heldOut: Movie[], scores: Scores, user: SyntheticUser): UserMetrics {
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

/** TensorFlow.js prints a one-off "use tfjs-node" banner; keep benchmark output clean. */
async function silently<T>(task: () => Promise<T>): Promise<T> {
  const warn = console.warn
  console.warn = () => {}
  try {
    return await task()
  } finally {
    console.warn = warn
  }
}

async function runIndividualBenchmark(world: SyntheticWorld, heldOut: SyntheticMovie[], labels: LabelPool, rng: () => number) {
  const catalog = world.movies
  const results = {} as Record<LabelCount, IndividualResult>

  for (const labelCount of LABEL_COUNTS) {
    const result: IndividualResult = {
      metrics: Object.fromEntries(METHODS.map((method) => [method, []])) as unknown as Record<Method, UserMetrics[]>,
      confidence: { mlp: [], gcn: [] },
      notReady: 0,
    }
    const started = Date.now()

    for (const user of world.users) {
      const labelled = labels.take(user, labelCount)
      const shipped = shippedProfile(user, catalog, labelled)
      const examples = buildNeuralExamples(catalog, shipped, [])
      const mlp = await silently(() => trainNeuralTaste(catalog, examples))
      const gcn = await silently(() => trainGraphNeuralTaste(catalog, examples))
      result.notReady += [mlp, gcn].filter(({ status }) => status !== 'ready').length
      result.confidence.mlp.push(mlp.confidence)
      result.confidence.gcn.push(gcn.confidence)

      // Key order matters: `random` draws from the seeded rng first.
      const scoresByMethod: Record<Method, Scores> = {
        random: Object.fromEntries(heldOut.map(({ id }) => [id, rng()])),
        popularity: Object.fromEntries(heldOut.map(({ id, rating }) => [id, rating])),
        scorer: scorerScores(heldOut, shipped),
        'scorer+feedback': scorerScores(heldOut, feedbackProfile(user, catalog, labelled)),
        mlp: mlp.predictions,
        gcn: gcn.predictions,
        'mlp-hybrid': hybridScores(heldOut, shipped, mlp),
        'gcn-hybrid': hybridScores(heldOut, shipped, gcn),
      }
      for (const method of METHODS) result.metrics[method].push(evaluateRanking(heldOut, scoresByMethod[method], user))
    }

    results[labelCount] = result
    process.stderr.write(`  ${labelCount} labels/user: ${world.users.length} users in ${((Date.now() - started) / 1000).toFixed(1)}s\n`)
  }
  return results
}

// ---------------------------------------------------------------------------
// Group fairness benchmark

interface GroupOutcome { mean: number; min: number; misery: number }

function outcomeFor(movie: Movie, members: SyntheticUser[]): GroupOutcome {
  const ratings = members.map((member) => member.ratings[movie.id])
  return { mean: mean(ratings), min: Math.min(...ratings), misery: ratings.some((rating) => rating <= MISERY) ? 1 : 0 }
}

const bestBy = (outcomes: GroupOutcome[], better: (next: GroupOutcome, best: GroupOutcome) => boolean) =>
  outcomes.reduce((best, next) => (better(next, best) ? next : best))

function runGroupBenchmark(config: Config, world: SyntheticWorld, heldOut: SyntheticMovie[], labels: LabelPool, rng: () => number) {
  const profiles = new Map(world.users.map((user) => [user.id, feedbackProfile(user, world.movies, labels.take(user, config.groupLabels))]))
  const rows: Record<string, GroupOutcome[]> = {
    random: [],
    ...Object.fromEntries(GROUP_STRATEGIES.map((strategy) => [strategy, []])),
    'oracle (max mean)': [],
    'oracle (max min)': [],
  }

  for (let triple = 0; triple < config.triples; triple += 1) {
    const members = shuffle(rng, world.users).slice(0, GROUP_SIZE)
    const memberProfiles = members.map(({ id }) => profiles.get(id)!)
    for (const strategy of GROUP_STRATEGIES) {
      const [top] = scoreCandidates(heldOut, memberProfiles, { rankingMode: strategy })
      rows[strategy].push(outcomeFor(top.movie, members))
    }
    rows.random.push(outcomeFor(heldOut[Math.floor(rng() * heldOut.length)], members))

    const outcomes = heldOut.map((movie) => outcomeFor(movie, members))
    rows['oracle (max mean)'].push(bestBy(outcomes, (next, best) => next.mean > best.mean))
    rows['oracle (max min)'].push(bestBy(outcomes, (next, best) => next.min > best.min || (next.min === best.min && next.mean > best.mean)))
  }
  return rows
}

// ---------------------------------------------------------------------------
// Report

const percent = (value: number) => (value * 100).toFixed(1)

function renderReport(
  config: Config,
  world: SyntheticWorld,
  heldOut: SyntheticMovie[],
  individual: Record<LabelCount, IndividualResult>,
  groups: Record<string, GroupOutcome[]>,
  ci: (values: number[]) => Interval,
) {
  const lines: string[] = []
  const json = {
    config,
    individual: {} as Record<number, { confidence: { mlp: number; gcn: number }; metrics: Record<string, Record<keyof UserMetrics, Interval>> }>,
    groups: {} as Record<string, Record<keyof GroupOutcome, Interval>>,
  }

  const heldOutRatings = world.users.flatMap((user) => heldOut.map(({ id }) => user.ratings[id]))
  const likedShare = heldOutRatings.filter((rating) => rating >= LIKED).length / heldOutRatings.length
  lines.push('# Recommendation benchmark', '')
  lines.push(`seed=${config.seed} · ${config.movies} movies · ${config.directors} directors · ${config.users} users · ${config.heldOut} held-out movies (fixed) · ${config.movies - config.heldOut} label pool`)
  lines.push(`held-out hidden ratings: mean ${mean(heldOutRatings).toFixed(2)}, ${percent(likedShare)}% liked (>= ${LIKED})`)
  lines.push(`NDCG@${K} (linear gain = hidden rating), Precision@${K} (liked >= ${LIKED}), Spearman over all held-out movies. Mean [95% bootstrap CI] across users.`, '')

  for (const labelCount of LABEL_COUNTS) {
    const { metrics, confidence, notReady } = individual[labelCount]
    const mlp = mean(confidence.mlp)
    const gcn = mean(confidence.gcn)
    lines.push(`## ${labelCount} labels per user`, '')
    lines.push(`MLP confidence ${mlp.toFixed(1)} (blend ${percent(confidenceToBlend(mlp))}%) · GCN confidence ${gcn.toFixed(1)} (blend ${percent(confidenceToBlend(gcn))}%) · untrained models: ${notReady}`, '')
    lines.push(`| Method | NDCG@${K} | Precision@${K} | Spearman |`, '| --- | --- | --- | --- |')

    const table: Record<string, Record<keyof UserMetrics, Interval>> = {}
    for (const method of METHODS) {
      const rows = metrics[method]
      const summary = {
        ndcg: ci(rows.map((row) => row.ndcg)),
        precision: ci(rows.map((row) => row.precision)),
        spearman: ci(rows.map((row) => row.spearman)),
      }
      table[method] = summary
      lines.push(`| ${method} | ${formatInterval(summary.ndcg)} | ${formatInterval(summary.precision)} | ${formatInterval(summary.spearman)} |`)
    }
    json.individual[labelCount] = { confidence: { mlp, gcn }, metrics: table }
    lines.push('')
  }

  lines.push(`## Group fairness (${config.triples} random triples, members use scorer+feedback profiles from ${config.groupLabels} labels)`, '')
  lines.push("Top pick scored on members' hidden ratings. Mean [95% bootstrap CI] across triples.", '')
  lines.push(`| Strategy | Mean member rating | Min member rating | Someone rates <= ${MISERY} |`, '| --- | --- | --- | --- |')
  for (const [strategy, rows] of Object.entries(groups)) {
    const summary = {
      mean: ci(rows.map((row) => row.mean)),
      min: ci(rows.map((row) => row.min)),
      misery: ci(rows.map((row) => row.misery * 100)),
    }
    json.groups[strategy] = summary
    lines.push(`| ${strategy} | ${formatInterval(summary.mean, 2)} | ${formatInterval(summary.min, 2)} | ${formatInterval(summary.misery, 1)}% |`)
  }

  return { markdown: lines.join('\n'), json }
}

// ---------------------------------------------------------------------------

async function main() {
  const config = parseArgs(process.argv.slice(2))
  const restoreMathRandom = patchMathRandom(config.seed ^ 0x9e3779b9)
  const rng = createRng(config.seed)
  const world = generateWorld(rng, config)
  const shuffled = shuffle(rng, world.movies)
  const heldOut = shuffled.slice(0, config.heldOut)
  const trainPool = shuffled.slice(config.heldOut)
  const labels = createLabelPool(new Map(world.users.map((user) => [user.id, shuffle(rng, trainPool)])))

  process.stderr.write(`Reel Circle recommendation benchmark (seed ${config.seed})\n`)
  const individual = await runIndividualBenchmark(world, heldOut, labels, rng)
  const groups = runGroupBenchmark(config, world, heldOut, labels, rng)
  restoreMathRandom()

  const ciRng = createRng(config.seed + 1)
  const { markdown, json } = renderReport(config, world, heldOut, individual, groups, (values) => bootstrapCI(values, ciRng))
  console.log(markdown)
  if (config.json) writeFileSync(config.json, `${JSON.stringify(json, null, 2)}\n`)
}

await main()
