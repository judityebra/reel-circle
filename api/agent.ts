import { createConversationPreferences, updateConversationPreferences, type ConversationPreferences } from '../src/lib/constraints.js'
import { explainPreferenceMatch, scoreCandidates, type Movie, type Profile } from '../src/lib/recommendation.js'

interface ServerlessRequest {
  method?: string
  body?: unknown
}

interface ServerlessResponse {
  status(code: number): ServerlessResponse
  json(body: unknown): void
  setHeader(name: string, value: string): void
}

interface AgentRequest {
  message: string
  preferences?: ConversationPreferences
  candidates?: Movie[]
  profiles?: Profile[]
  limit?: number
}

interface AgentStep {
  tool: 'extract_preferences' | 'filter_candidates' | 'rank_candidates' | 'validate_results'
  status: 'completed' | 'skipped'
  durationMs: number
  observation: string
}

interface ModelExtraction {
  genres?: string[]
  excludedGenres?: string[]
  platforms?: string[]
  moods?: string[]
  maxRuntime?: number
  yearRange?: [number, number]
  languages?: string[]
  peopleCount?: number
  viewingContext?: string
  clarificationQuestion?: string
}

const allowedGenres = ['Action', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']
const allowedMoods = ['Gentle', 'Playful', 'Electric', 'Reflective', 'Twisty']

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value)
const stringList = (value: unknown, allowed?: string[]) => Array.isArray(value)
  ? [...new Set(value.filter((item): item is string => typeof item === 'string' && (!allowed || allowed.includes(item))))]
  : []

function parseRequest(body: unknown): AgentRequest | undefined {
  if (!isRecord(body) || typeof body.message !== 'string') return undefined
  const message = body.message.trim()
  if (!message || message.length > 1_000) return undefined
  return {
    message,
    preferences: isRecord(body.preferences) ? body.preferences as unknown as ConversationPreferences : undefined,
    candidates: Array.isArray(body.candidates) ? body.candidates.slice(0, 250) as Movie[] : [],
    profiles: Array.isArray(body.profiles) ? body.profiles.slice(0, 12) as Profile[] : [],
    limit: typeof body.limit === 'number' ? Math.max(1, Math.min(10, Math.floor(body.limit))) : 5,
  }
}

function sanitizeExtraction(value: unknown): ModelExtraction | undefined {
  if (!isRecord(value)) return undefined
  const yearRange = Array.isArray(value.yearRange) && value.yearRange.length === 2 && value.yearRange.every((year) => typeof year === 'number')
    ? value.yearRange as [number, number]
    : undefined
  return {
    genres: stringList(value.genres, allowedGenres),
    excludedGenres: stringList(value.excludedGenres, allowedGenres),
    platforms: stringList(value.platforms),
    moods: stringList(value.moods, allowedMoods),
    maxRuntime: typeof value.maxRuntime === 'number' && value.maxRuntime > 0 ? Math.min(600, value.maxRuntime) : undefined,
    yearRange,
    languages: stringList(value.languages),
    peopleCount: typeof value.peopleCount === 'number' ? Math.max(1, Math.min(20, Math.floor(value.peopleCount))) : undefined,
    viewingContext: typeof value.viewingContext === 'string' ? value.viewingContext.slice(0, 80) : undefined,
    clarificationQuestion: typeof value.clarificationQuestion === 'string' ? value.clarificationQuestion.slice(0, 240) : undefined,
  }
}

async function extractWithModel(message: string, current: ConversationPreferences): Promise<ModelExtraction | undefined> {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) return undefined
  const baseUrl = (process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1').replace(/\/$/, '')
  const requestModel = async (responseFormat: Record<string, unknown>) => fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL ?? 'gpt-4.1-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: 'Return one JSON object containing only movie-viewing preferences from the latest turn. Preserve relevant current preferences. Keys: genres, excludedGenres, platforms, moods, maxRuntime, yearRange, languages, peopleCount, viewingContext, clarificationQuestion. Use canonical English TMDB genre names and ISO 639-1 language codes. Use empty arrays or null for unknown values.' },
        { role: 'user', content: JSON.stringify({ current, latestTurn: message }) },
      ],
      response_format: responseFormat,
    }),
  })
  let response = await requestModel({
        type: 'json_schema',
        json_schema: {
          name: 'movie_preferences',
          strict: true,
          schema: {
            type: 'object',
            additionalProperties: false,
            properties: {
              genres: { type: 'array', items: { type: 'string', enum: allowedGenres } },
              excludedGenres: { type: 'array', items: { type: 'string', enum: allowedGenres } },
              platforms: { type: 'array', items: { type: 'string' } },
              moods: { type: 'array', items: { type: 'string', enum: allowedMoods } },
              maxRuntime: { type: ['number', 'null'] },
              yearRange: { type: ['array', 'null'], items: { type: 'number' }, minItems: 2, maxItems: 2 },
              languages: { type: 'array', items: { type: 'string' } },
              peopleCount: { type: ['number', 'null'] },
              viewingContext: { type: ['string', 'null'] },
              clarificationQuestion: { type: ['string', 'null'] },
            },
            required: ['genres', 'excludedGenres', 'platforms', 'moods', 'maxRuntime', 'yearRange', 'languages', 'peopleCount', 'viewingContext', 'clarificationQuestion'],
          },
        },
  })
  if (!response.ok && (response.status === 400 || response.status === 422)) {
    response = await requestModel({ type: 'json_object' })
  }
  if (!response.ok) throw new Error(`Model request failed: ${response.status}`)
  const payload = await response.json() as { choices?: Array<{ message?: { content?: string } }> }
  const content = payload.choices?.[0]?.message?.content
  return content ? sanitizeExtraction(JSON.parse(content)) : undefined
}

function mergeModelExtraction(current: ConversationPreferences, extraction: ModelExtraction, now: number): ConversationPreferences {
  return {
    ...current,
    genres: extraction.genres?.length ? extraction.genres : current.genres,
    excludedGenres: extraction.excludedGenres?.length ? extraction.excludedGenres : current.excludedGenres,
    platforms: extraction.platforms?.length ? extraction.platforms : current.platforms,
    moods: extraction.moods?.length ? extraction.moods : current.moods,
    maxRuntime: extraction.maxRuntime ?? current.maxRuntime,
    yearRange: extraction.yearRange ?? current.yearRange,
    languages: extraction.languages?.length ? extraction.languages : current.languages,
    peopleCount: extraction.peopleCount ?? current.peopleCount,
    viewingContext: extraction.viewingContext ?? current.viewingContext,
    lastUpdatedAt: now,
  }
}

export default async function handler(request: ServerlessRequest, response: ServerlessResponse) {
  if (request.method !== 'POST') return response.status(405).json({ error: 'Method not allowed' })
  const input = parseRequest(request.body)
  if (!input) return response.status(400).json({ error: 'A message of 1-1000 characters is required' })

  const startedAt = Date.now()
  const trace: AgentStep[] = []
  const current = input.preferences ?? createConversationPreferences(startedAt)
  let modelExtraction: ModelExtraction | undefined
  let extractionMode: 'model' | 'deterministic' = 'deterministic'
  let modelStatus: 'not-configured' | 'failed' | 'succeeded' = process.env.OPENAI_API_KEY ? 'failed' : 'not-configured'
  let modelError: string | undefined
  const extractionStartedAt = Date.now()
  try {
    modelExtraction = await extractWithModel(input.message, current)
    if (modelExtraction) {
      extractionMode = 'model'
      modelStatus = 'succeeded'
    }
  } catch (error) {
    modelExtraction = undefined
    const status = error instanceof Error ? error.message.match(/Model request failed: (\d{3})/)?.[1] : undefined
    modelError = status ? `http-${status}` : error instanceof SyntaxError ? 'invalid-json' : 'request-error'
  }
  const preferences = modelExtraction
    ? mergeModelExtraction(current, modelExtraction, Date.now())
    : updateConversationPreferences(current, input.message, Date.now())
  trace.push({ tool: 'extract_preferences', status: 'completed', durationMs: Date.now() - extractionStartedAt, observation: `${extractionMode} structured extraction` })

  const filterStartedAt = Date.now()
  const ranked = scoreCandidates(input.candidates ?? [], input.profiles ?? [], preferences)
  trace.push({ tool: 'filter_candidates', status: input.candidates?.length ? 'completed' : 'skipped', durationMs: Date.now() - filterStartedAt, observation: `${ranked.length}/${input.candidates?.length ?? 0} candidates satisfy hard constraints` })

  const recommendations = ranked.slice(0, input.limit).map(({ movie, score, reasons }) => ({
    movie,
    score,
    matchedPreferences: reasons,
    explanation: explainPreferenceMatch(movie, preferences),
  }))
  trace.push({ tool: 'rank_candidates', status: recommendations.length ? 'completed' : 'skipped', durationMs: 0, observation: `${recommendations.length} recommendations returned` })

  const violations = recommendations.filter(({ movie }) =>
    preferences.excludedGenres.some((genre) => movie.genres.includes(genre)) ||
    Boolean(preferences.maxRuntime && movie.runtime > preferences.maxRuntime),
  ).length
  trace.push({ tool: 'validate_results', status: 'completed', durationMs: 0, observation: `${violations} hard-constraint violations` })

  response.setHeader('Cache-Control', 'no-store')
  return response.status(200).json({
    preferences,
    recommendations,
    clarificationQuestion: modelExtraction?.clarificationQuestion,
    meta: { extractionMode, modelStatus, modelError, durationMs: Date.now() - startedAt, trace },
  })
}
