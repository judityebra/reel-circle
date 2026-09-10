import { useEffect, useMemo, useRef, useState, type CSSProperties, type SyntheticEvent } from 'react'
import { Background, Controls, ReactFlow, useNodesState, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { Activity, BarChart3, Bot, BrainCircuit, Check, CheckCircle2, ChevronRight, Clock3, Copy, Cpu, Database, Download, Film, FlaskConical, GitCompareArrows, Heart, Layers3, LoaderCircle, MessageSquareText, MonitorPlay, Moon, Network, Play, Plus, Route, RotateCcw, Search, Share2, ShieldCheck, Shuffle, Sparkles, Sun, Target, Thermometer, ThumbsDown, Trash2, Upload, Users, X } from 'lucide-react'
import { catalog as demoCatalog, demoProfiles, profileColors } from './data'
import { evaluateAgentTrace, scoreAgentRun } from './lib/agent-evaluation'
import { describeConstraints, parseConstraintPrompt } from './lib/constraints'
import { runDecisionAgent, type AgentTraceEntry } from './lib/decision-agent'
import { compareRankings, explainRecommendationPath, getDisagreement, type ExperimentRecord } from './lib/experiments'
import { trainGraphNeuralTaste, type GraphNeuralResult } from './lib/graph-neural'
import { parseLetterboxdExport, parseLetterboxdZip } from './lib/letterboxd'
import { summarizeFeedback } from './lib/metrics'
import { fetchMovieCatalog, searchMovieCatalog } from './lib/movie-api'
import { buildNeuralExamples, trainNeuralTaste, type NeuralTasteResult } from './lib/neural'
import { resolveLocale, translate, type Locale } from './lib/i18n'
import { applyComparisonFeedback, applyTasteFeedback, chooseTasteQuestion, getMovieFeatures, scoreCandidates, type Movie, type Profile, type RankingMode, type TasteFeedback } from './lib/recommendation'
import { clearAllLocalData, decodeSharedRoom, encodeSharedRoom, loadCachedCatalog, loadSession, removeSeedProfiles, saveCachedCatalog, saveSession, type FeedbackEvent, type SessionSnapshot } from './lib/session'
import { resolveTheme, type ThemePreference } from './lib/theme'
import './App.css'

const genres = ['Drama', 'Comedy', 'Mystery', 'Romance', 'Sci-Fi']
const moods = ['Gentle', 'Playful', 'Electric', 'Reflective', 'Twisty']
const platforms = ['Netflix', 'Prime Video', 'Movistar Plus+', 'Filmin', 'MUBI', 'Max', 'Disney+', 'SkyShowtime']
const catalogPageCount = Number(import.meta.env.VITE_MOVIE_CATALOG_PAGES) || 10
const fallbackPoster = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450"%3E%3Crect width="300" height="450" fill="%2320221f"/%3E%3Ccircle cx="150" cy="205" r="72" fill="none" stroke="%23087d67" stroke-width="18"/%3E%3Ccircle cx="150" cy="205" r="28" fill="%23f15b2a"/%3E%3Ctext x="150" y="350" text-anchor="middle" fill="white" font-family="sans-serif" font-size="24" letter-spacing="4"%3EREEL CIRCLE%3C/text%3E%3C/svg%3E'

const usePosterFallback = (event: SyntheticEvent<HTMLImageElement>) => {
  event.currentTarget.src = fallbackPoster
}

const displayValue = (locale: Locale, value: string) => {
  if (locale === 'en') return value
  const runtime = value.match(/^Under (\d+) min$/)
  if (runtime) return `Menos de ${runtime[1]} min`
  return ({
    Comedy: 'Comedia', Drama: 'Drama', Mystery: 'Misterio', Romance: 'Romance', 'Sci-Fi': 'Ciencia ficción', Horror: 'Terror', Thriller: 'Suspense', Action: 'Acción', Fantasy: 'Fantasía', Animation: 'Animación', Documentary: 'Documental', Crime: 'Crimen', Gentle: 'Suave', Playful: 'Divertida', Electric: 'Intensa', Reflective: 'Reflexiva', Twisty: 'Enrevesada', 'Any movie': 'Cualquier película', 'Nobody hates it': 'Que nadie la odie', 'Balanced group pick': 'Elección equilibrada',
  }[value] ?? value)
}

const localizeReason = (locale: Locale, reason: string) => {
  if (locale === 'en') return reason
  if (reason.startsWith('Shared taste: ')) return `Gusto compartido: ${reason.replace('Shared taste: ', '').split(', ').map((genre) => displayValue(locale, genre)).join(', ')}`
  if (reason.endsWith(' bridges this group')) return `${displayValue(locale, reason.replace(' bridges this group', ''))} conecta los gustos del grupo`
  const quick = reason.match(/^Quick (\d+)-minute watch$/)
  if (quick) return `Sesión rápida de ${quick[1]} min`
  const minutes = reason.match(/^(\d+) minutes$/)
  if (minutes) return `${minutes[1]} minutos`
  if (reason.endsWith(' community rating')) return `${reason.replace(' community rating', '')} de valoración comunitaria`
  if (reason === 'Neural taste model included') return 'Modelo neuronal de gustos incluido'
  return reason
}

function App() {
  const [movieCatalog, setMovieCatalog] = useState<Movie[]>(demoCatalog)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [maxRuntime, setMaxRuntime] = useState<number | undefined>(120)
  const [activeGenres, setActiveGenres] = useState<string[]>([])
  const [activePlatforms, setActivePlatforms] = useState<string[]>([])
  const [activeMoods, setActiveMoods] = useState<string[]>([])
  const [rankingMode, setRankingMode] = useState<RankingMode>('balanced')
  const [constraintPrompt, setConstraintPrompt] = useState('')
  const [promptInterpretation, setPromptInterpretation] = useState<string[] | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [activeProfileId, setActiveProfileId] = useState('')
  const [feedbackNotice, setFeedbackNotice] = useState('')
  const [feedbackEvents, setFeedbackEvents] = useState<FeedbackEvent[]>([])
  const [tasteQuestion, setTasteQuestion] = useState<[Movie, Movie] | null>(null)
  const [showDeepSignals, setShowDeepSignals] = useState(true)
  const [graphMode, setGraphMode] = useState<'taste' | 'neural' | 'gnn'>('taste')
  const [activeModel, setActiveModel] = useState<'mlp' | 'gcn'>('mlp')
  const [neuralEnabled, setNeuralEnabled] = useState(true)
  const [neuralTraining, setNeuralTraining] = useState(false)
  const [neuralResults, setNeuralResults] = useState<Record<string, NeuralTasteResult>>({})
  const [graphNeuralResults, setGraphNeuralResults] = useState<Record<string, GraphNeuralResult>>({})
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([])
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [labMode, setLabMode] = useState(false)
  const [agentGoal, setAgentGoal] = useState('Pick a fair movie under two hours')
  const [agentTrace, setAgentTrace] = useState<AgentTraceEntry[]>([])
  const [agentStatus, setAgentStatus] = useState<'idle' | 'running' | 'awaiting-input' | 'awaiting-approval'>('idle')
  const [themePreference, setThemePreference] = useState<ThemePreference>(() => {
    const stored = localStorage.getItem('reel-circle-theme')
    return stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system'
  })
  const [locale, setLocale] = useState<Locale>(() => resolveLocale(localStorage.getItem('reel-circle-locale'), navigator.language))
  const [showOnboarding, setShowOnboarding] = useState(() => localStorage.getItem('reel-circle-onboarding') !== 'done')
  const t = (key: Parameters<typeof translate>[1]) => translate(locale, key)
  const [hydrated, setHydrated] = useState(false)
  const [catalogSource, setCatalogSource] = useState<'curated' | 'cached' | 'live'>('curated')
  const [catalogLoading, setCatalogLoading] = useState(Boolean(import.meta.env.VITE_MOVIE_API_URL))
  const [catalogProgress, setCatalogProgress] = useState({ loadedPages: 0, requestedPages: catalogPageCount, movies: 0 })
  const [catalogError, setCatalogError] = useState('')
  const [catalogQuery, setCatalogQuery] = useState('')
  const [catalogSearching, setCatalogSearching] = useState(false)
  const [restoredPositions, setRestoredPositions] = useState<Record<string, { x: number; y: number }>>({})
  const fileInput = useRef<HTMLInputElement>(null)
  const comparisonDialog = useRef<HTMLElement>(null)
  const comparisonOpener = useRef<HTMLElement | null>(null)
  const activeModelResults = activeModel === 'mlp' ? neuralResults : graphNeuralResults
  const neuralScores = useMemo(
    () => Object.fromEntries(Object.entries(activeModelResults).filter(([, result]) => result.status === 'ready').map(([profileId, result]) => [profileId, result.predictions])),
    [activeModelResults],
  )
  const neuralBlend = useMemo(() => {
    const readyResults = Object.values(activeModelResults).filter(({ status }) => status === 'ready')
    if (readyResults.length === 0) return 0
    const averageConfidence = readyResults.reduce((total, result) => total + result.confidence, 0) / readyResults.length
    return Math.min(0.5, averageConfidence / 200)
  }, [activeModelResults])
  const baselineRecommendations = useMemo(
    () => scoreCandidates(movieCatalog, profiles, { maxRuntime, genres: activeGenres, platforms: activePlatforms, moods: activeMoods, rankingMode }),
    [movieCatalog, profiles, maxRuntime, activeGenres, activePlatforms, activeMoods, rankingMode],
  )
  const recommendations = useMemo(
    () => scoreCandidates(movieCatalog, profiles, { maxRuntime, genres: activeGenres, platforms: activePlatforms, moods: activeMoods, rankingMode, neuralScores: neuralEnabled ? neuralScores : undefined, neuralBlend: neuralEnabled ? neuralBlend : 0 }),
    [movieCatalog, profiles, maxRuntime, activeGenres, activePlatforms, activeMoods, rankingMode, neuralEnabled, neuralScores, neuralBlend],
  )
  const selected = recommendations.find(({ movie }) => movie.id === selectedId) ?? recommendations[0]
  const activeProfile = profiles.find(({ id }) => id === activeProfileId) ?? profiles[0]
  const metrics = useMemo(() => summarizeFeedback(feedbackEvents), [feedbackEvents])
  const agentEvaluation = useMemo(() => evaluateAgentTrace(agentTrace), [agentTrace])
  const latestAgentRun = useMemo(() => {
    const runId = agentTrace[0]?.runId
    if (runId) return agentTrace.filter((entry) => entry.runId === runId)
    const nextRunStart = agentTrace.slice(1).findIndex(({ step }) => step === 1)
    return nextRunStart < 0 ? agentTrace : agentTrace.slice(0, nextRunStart + 1)
  }, [agentTrace])
  const latestAgentScore = useMemo(() => scoreAgentRun(latestAgentRun), [latestAgentRun])
  const activeNeuralExamples = useMemo(
    () => activeProfile ? buildNeuralExamples(movieCatalog, activeProfile, feedbackEvents) : [],
    [movieCatalog, activeProfile, feedbackEvents],
  )
  const activeNeuralResult = activeProfile ? neuralResults[activeProfile.id] : undefined
  const activeGraphNeuralResult = activeProfile ? graphNeuralResults[activeProfile.id] : undefined
  const activeModelResult = activeModel === 'mlp' ? activeNeuralResult : activeGraphNeuralResult
  const rankingComparison = useMemo(() => compareRankings(baselineRecommendations, recommendations), [baselineRecommendations, recommendations])
  const selectedDisagreement = selected ? getDisagreement(selected) : 0
  const selectedPath = activeProfile && selected ? explainRecommendationPath(activeProfile, selected.movie) : undefined
  const selectedActivations = selected && activeNeuralResult?.status === 'ready' ? activeNeuralResult.activations?.[selected.movie.id] : undefined
  const featureSignals = useMemo(
    () => Object.entries(activeProfile?.featureWeights ?? {}).sort((left, right) => Math.abs(right[1]) - Math.abs(left[1])).slice(0, 4),
    [activeProfile],
  )

  const graphNodes: Node[] = useMemo(() => [
    ...profiles.map((profile, index) => ({
      id: profile.id,
      position: restoredPositions[profile.id] ?? { x: 30, y: 55 + index * 105 },
      initialWidth: 150,
      initialHeight: 48,
      data: { label: `${profile.name}\n${profile.watched.length} watched` },
      className: profile.id === activeProfile?.id ? 'person-node selected-person-node' : 'person-node',
      style: { borderColor: profile.color },
    })),
    ...genres.map((genre, index) => {
      const learnedWeight = activeProfile?.tasteWeights?.[genre] ?? 0
      return {
      id: `genre-${genre}`,
      position: restoredPositions[`genre-${genre}`] ?? { x: showDeepSignals ? 210 : 265, y: 12 + index * 68 },
      initialWidth: 68,
      initialHeight: 68,
      data: { label: learnedWeight ? `${displayValue(locale, genre)}\n${learnedWeight > 0 ? '+' : ''}${learnedWeight.toFixed(2)}` : displayValue(locale, genre) },
      className: activeGenres.includes(genre) ? 'genre-node selected-genre-node' : 'genre-node',
    }}),
    ...(showDeepSignals ? featureSignals.map(([feature, weight], index) => ({
      id: `feature-${feature}`,
      position: restoredPositions[`feature-${feature}`] ?? { x: 380, y: 35 + index * 82 },
      initialWidth: 120,
      initialHeight: 44,
      data: { label: `${feature.replace(':', '\n')}\n${weight > 0 ? '+' : ''}${weight.toFixed(2)}` },
      className: weight < 0 ? 'feature-node negative-feature-node' : 'feature-node',
    })) : []),
    ...recommendations.slice(0, 3).map((item, index) => ({
      id: item.movie.id,
      position: restoredPositions[item.movie.id] ?? { x: showDeepSignals ? 550 : 485, y: 30 + index * 102 },
      initialWidth: 150,
      initialHeight: 48,
      data: { label: `${item.movie.title}\n${item.score}% match` },
      className: item.movie.id === selected?.movie.id ? 'movie-node selected-node' : 'movie-node',
      style: showHeatmap ? { background: `color-mix(in srgb, #ef6535 ${Math.min(55, getDisagreement(item))}%, #fff)` } : undefined,
    })),
  ], [profiles, activeProfile, activeGenres, featureSignals, showDeepSignals, showHeatmap, recommendations, selected, restoredPositions, locale])
  const graphEdges: Edge[] = useMemo(() => [
    ...profiles.flatMap((profile) => genres.filter((genre) => profile.preferredGenres.includes(genre) || profile.tasteWeights?.[genre]).map((genre) => {
      const weight = profile.tasteWeights?.[genre] ?? 0
      return { id: `${profile.id}-${genre}`, source: profile.id, target: `genre-${genre}`, style: { stroke: profile.color, opacity: weight < 0 ? 0.2 : 0.55, strokeWidth: 1 + Math.max(0, weight), strokeDasharray: weight < 0 ? '4 4' : undefined } }
    })),
    ...recommendations.slice(0, 3).flatMap((item) => item.movie.genres.map((genre) => ({ id: `${genre}-${item.movie.id}`, source: `genre-${genre}`, target: item.movie.id, animated: item.movie.id === selected?.movie.id, style: { stroke: '#b8b3a8' } }))),
    ...(showDeepSignals && activeProfile ? featureSignals.flatMap(([feature, weight]) => [
      { id: `${activeProfile.id}-${feature}`, source: activeProfile.id, target: `feature-${feature}`, style: { stroke: activeProfile.color, opacity: 0.45, strokeDasharray: weight < 0 ? '4 4' : undefined } },
      ...recommendations.slice(0, 3).filter(({ movie }) => getMovieFeatures(movie).includes(feature)).map(({ movie }) => ({ id: `${feature}-${movie.id}`, source: `feature-${feature}`, target: movie.id, style: { stroke: '#7fa698', opacity: 0.55 } })),
    ]) : []),
  ], [profiles, recommendations, selected, showDeepSignals, activeProfile, featureSignals])
  const neuralGraphNodes: Node[] = useMemo(() => {
    const inputs = ['Feature hash\n32 buckets', 'Runtime\nnormalized', 'Rating\nnormalized']
    const hiddenOne = Array.from({ length: 6 }, (_, index) => `H1 · ${index + 1}`)
    const hiddenTwo = Array.from({ length: 4 }, (_, index) => `H2 · ${index + 1}`)
    return [
      ...inputs.map((label, index) => ({ id: `nn-input-${index}`, position: { x: 35, y: 65 + index * 105 }, data: { label }, className: 'nn-node nn-input-node', initialWidth: 110, initialHeight: 48 })),
      ...hiddenOne.map((label, index) => { const activation = selectedActivations?.hiddenOne[index] ?? 0; return { id: `nn-h1-${index}`, position: { x: 255, y: 20 + index * 67 }, data: { label: `${label}\n${activation.toFixed(2)}` }, className: 'nn-node nn-hidden-node', initialWidth: 62, initialHeight: 42, style: { opacity: activeNeuralResult?.status === 'ready' ? Math.max(0.25, Math.min(1, activation + 0.25)) : 0.55 } } }),
      ...hiddenTwo.map((label, index) => { const activation = selectedActivations?.hiddenTwo[index] ?? 0; return { id: `nn-h2-${index}`, position: { x: 430, y: 75 + index * 82 }, data: { label: `${label}\n${activation.toFixed(2)}` }, className: 'nn-node nn-hidden-node', initialWidth: 62, initialHeight: 42, style: { opacity: activeNeuralResult?.status === 'ready' ? Math.max(0.25, Math.min(1, activation + 0.25)) : 0.55 } } }),
      { id: 'nn-output', position: { x: 590, y: 180 }, data: { label: activeNeuralResult?.status === 'ready' ? `Preference\n${activeNeuralResult.confidence}% confidence` : 'Preference\nnot trained' }, className: 'nn-node nn-output-node', initialWidth: 120, initialHeight: 54 },
    ]
  }, [activeNeuralResult, selectedActivations])
  const neuralGraphEdges: Edge[] = useMemo(() => [
    ...Array.from({ length: 3 }, (_, input) => Array.from({ length: 6 }, (_, hidden) => ({ id: `nn-i${input}-h1${hidden}`, source: `nn-input-${input}`, target: `nn-h1-${hidden}`, style: { stroke: '#85a99b', opacity: 0.32 } }))).flat(),
    ...Array.from({ length: 6 }, (_, first) => Array.from({ length: 4 }, (_, second) => ({ id: `nn-h1${first}-h2${second}`, source: `nn-h1-${first}`, target: `nn-h2-${second}`, style: { stroke: '#7b9db0', opacity: 0.32 } }))).flat(),
    ...Array.from({ length: 4 }, (_, second) => ({ id: `nn-h2${second}-out`, source: `nn-h2-${second}`, target: 'nn-output', animated: activeNeuralResult?.status === 'ready', style: { stroke: '#ef6535', opacity: 0.65 } })),
  ], [activeNeuralResult])
  const graphNeuralNodes: Node[] = useMemo(() => {
    const visibleMovies = recommendations.slice(0, 5).map(({ movie }) => movie)
    const visibleFeatures = [...new Set(visibleMovies.flatMap((movie) => movie.genres.map((genre) => `genre:${genre}`)))].slice(0, 6)
    return [
      ...visibleFeatures.map((feature, index) => ({ id: `gcn-${feature}`, position: { x: 65, y: 25 + index * 68 }, data: { label: feature.replace(':', '\n') }, className: 'gcn-node gcn-feature-node', initialWidth: 95, initialHeight: 44 })),
      ...visibleMovies.map((movie, index) => {
        const embedding = activeGraphNeuralResult?.embeddings[movie.id] ?? []
        const magnitude = embedding.length ? Math.sqrt(embedding.reduce((total, value) => total + value * value, 0)) : 0
        return { id: `gcn-movie-${movie.id}`, position: { x: 460, y: 40 + index * 78 }, data: { label: `${movie.title}\nembedding |h| ${magnitude.toFixed(2)}` }, className: 'gcn-node gcn-movie-node', initialWidth: 155, initialHeight: 50, style: { boxShadow: `0 0 0 ${Math.min(12, magnitude * 4)}px #397a9e20` } }
      }),
    ]
  }, [recommendations, activeGraphNeuralResult])
  const graphNeuralEdges: Edge[] = useMemo(() => recommendations.slice(0, 5).flatMap(({ movie }) => movie.genres.map((genre) => ({ id: `gcn-${genre}-${movie.id}`, source: `gcn-genre:${genre}`, target: `gcn-movie-${movie.id}`, animated: activeGraphNeuralResult?.status === 'ready', style: { stroke: '#397a9e', opacity: 0.5 } }))), [recommendations, activeGraphNeuralResult])
  const [renderedNodes, setRenderedNodes, onNodesChange] = useNodesState(graphNodes)

  useEffect(() => {
    const hydrate = async () => {
      try {
        const room = new URLSearchParams(window.location.search).get('room')
        if (room) {
          const shared = decodeSharedRoom(room)
          setProfiles(shared.profiles.map((profile) => ({ ...profile, ratings: [], watched: [] })))
          setMaxRuntime(shared.maxRuntime)
          setActiveGenres(shared.activeGenres)
          setActivePlatforms(shared.activePlatforms)
          setActiveMoods(shared.activeMoods)
          setRankingMode(shared.rankingMode)
          setActiveProfileId(shared.profiles[0]?.id ?? '')
          setFeedbackNotice('Private room preferences loaded')
        } else {
          const stored = await loadSession()
          if (stored) {
            const migratedProfiles = removeSeedProfiles(stored.profiles)
            const restoredProfiles = migratedProfiles.length ? migratedProfiles : demoProfiles
            setProfiles(restoredProfiles)
            setMaxRuntime(stored.maxRuntime)
            setActiveGenres(stored.activeGenres)
            setActivePlatforms(stored.activePlatforms)
            setActiveMoods(stored.activeMoods)
            setRankingMode(stored.rankingMode)
            setActiveProfileId(restoredProfiles.some(({ id }) => id === stored.activeProfileId) ? stored.activeProfileId : restoredProfiles[0].id)
            setFeedbackEvents(stored.feedbackEvents)
            setNeuralResults(stored.neuralResults ?? {})
            setGraphNeuralResults(stored.graphNeuralResults ?? {})
            setExperiments(stored.experiments ?? [])
            setAgentTrace(stored.agentTrace ?? [])
            setRestoredPositions(stored.nodePositions)
          } else {
            setProfiles(demoProfiles)
            setActiveProfileId(demoProfiles[0].id)
          }
        }
      } catch {
        setFeedbackNotice('Could not load the saved room')
      } finally {
        setHydrated(true)
      }
    }
    void hydrate()
  }, [])

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_MOVIE_API_URL as string | undefined
    if (!apiUrl) return
    let cancelled = false
    const loadCatalog = async () => {
      const cached = await loadCachedCatalog()
      if (!cancelled && cached?.length) {
        setMovieCatalog(cached)
        setCatalogSource('cached')
        setCatalogProgress({ loadedPages: catalogPageCount, requestedPages: catalogPageCount, movies: cached.length })
      }
      try {
        const movies = await fetchMovieCatalog(apiUrl, fetch, {
          pages: catalogPageCount,
          onProgress: (progress) => { if (!cancelled) setCatalogProgress(progress) },
        })
        if (cancelled) return
        if (movies.length > 0) {
          setMovieCatalog(movies)
          setCatalogSource('live')
          setCatalogError('')
          await saveCachedCatalog(movies)
        }
      } catch {
        if (!cancelled) {
          setCatalogError(cached?.length ? 'Live refresh failed; using cached movies' : 'Live catalog unavailable; using curated movies')
          setFeedbackNotice(cached?.length ? 'Live refresh failed; using cached movies' : 'Live catalog unavailable; using curated data')
        }
      } finally {
        if (!cancelled) setCatalogLoading(false)
      }
    }
    void loadCatalog()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    setRenderedNodes((current) => graphNodes.map((node) => {
      const existing = current.find(({ id }) => id === node.id)
      return {
        ...node,
        position: existing?.position ?? node.position,
        measured: existing?.measured,
      }
    }))
  }, [graphNodes, setRenderedNodes])

  useEffect(() => {
    if (!hydrated) return
    void saveSession({ profiles, maxRuntime, activeGenres, activePlatforms, activeMoods, rankingMode, activeProfileId, nodePositions: restoredPositions, feedbackEvents, neuralResults, graphNeuralResults, experiments, agentTrace })
  }, [hydrated, profiles, maxRuntime, activeGenres, activePlatforms, activeMoods, rankingMode, activeProfileId, restoredPositions, feedbackEvents, neuralResults, graphNeuralResults, experiments, agentTrace])

  useEffect(() => {
    if (!tasteQuestion) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setTasteQuestion(null)
      if (event.key !== 'Tab' || !comparisonDialog.current) return
      const focusable = [...comparisonDialog.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])')]
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable.at(-1)!
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('keydown', closeOnEscape)
      comparisonOpener.current?.focus()
    }
  }, [tasteQuestion])

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const applyTheme = () => {
      document.documentElement.dataset.theme = resolveTheme(themePreference, media.matches)
      document.documentElement.style.colorScheme = resolveTheme(themePreference, media.matches)
    }
    applyTheme()
    localStorage.setItem('reel-circle-theme', themePreference)
    media.addEventListener('change', applyTheme)
    return () => media.removeEventListener('change', applyTheme)
  }, [themePreference])

  useEffect(() => {
    document.documentElement.lang = locale
    localStorage.setItem('reel-circle-locale', locale)
  }, [locale])

  const importProfile = async (file?: File) => {
    if (!file) return
    try {
      const data = new Uint8Array(await file.arrayBuffer())
      const isZip = data[0] === 0x50 && data[1] === 0x4b
      const content = isZip ? '' : new TextDecoder().decode(data)
      const isRss = !isZip && (content.trimStart().startsWith('<?xml') || /^<rss[\s>]/i.test(content.trimStart()))
      const imported = isZip ? await parseLetterboxdZip(data) : parseLetterboxdExport(content)
      const ratedGenres = movieCatalog
        .filter((movie) => imported.ratings.some((rating) => rating.title.toLowerCase() === movie.title.toLowerCase()))
        .flatMap((movie) => movie.genres)
      const preferredGenres = [...new Set(ratedGenres)].slice(0, 3)
      const fileName = file.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ')
      const name = imported.profileName ?? fileName
      const existing = profiles.find((profile) => profile.name.toLocaleLowerCase() === name.toLocaleLowerCase())
      const profileId = existing?.id ?? `${name}-${Date.now()}`
      const importedProfile: Profile = {
        id: profileId,
        name: name || `Person ${profiles.length + 1}`,
        color: existing?.color ?? profileColors[profiles.length % profileColors.length],
        ratings: imported.ratings,
        watched: imported.watched,
        preferredGenres: preferredGenres.length ? preferredGenres : genres.slice(profiles.length % 3, profiles.length % 3 + 2),
      }
      setProfiles((current) => existing
        ? current.map((profile) => profile.id === existing.id ? importedProfile : profile)
        : [...current, importedProfile])
      setActiveProfileId(profileId)
      setFeedbackNotice(`Imported ${imported.ratings.length} ratings and ${imported.watched.length} watched titles for ${name}${isZip ? ' · complete export' : isRss ? ' · RSS contains recent activity only' : ''}`)
    } catch (error) {
      setFeedbackNotice(error instanceof Error ? error.message : 'Could not import this Letterboxd file')
    }
  }

  const toggleGenre = (genre: string) => {
    setActiveGenres((current) => current.includes(genre) ? current.filter((item) => item !== genre) : [...current, genre])
  }

  const togglePlatform = (platform: string) => {
    setActivePlatforms((current) => current.includes(platform) ? current.filter((item) => item !== platform) : [...current, platform])
  }

  const toggleMood = (mood: string) => {
    setActiveMoods((current) => current.includes(mood) ? current.filter((item) => item !== mood) : [...current, mood])
  }

  const mergeCatalog = (current: Movie[], incoming: Movie[]) => {
    const moviesById = new Map(current.map((movie) => [movie.id, movie]))
    incoming.forEach((movie) => moviesById.set(movie.id, movie))
    return [...moviesById.values()]
  }

  const loadMoreCatalog = async () => {
    const apiUrl = import.meta.env.VITE_MOVIE_API_URL as string | undefined
    if (!apiUrl || catalogLoading) return
    const startPage = catalogProgress.loadedPages + 1
    setCatalogLoading(true)
    try {
      const movies = await fetchMovieCatalog(apiUrl, fetch, {
        pages: 5,
        startPage,
        onProgress: (progress) => setCatalogProgress({
          loadedPages: startPage - 1 + progress.loadedPages,
          requestedPages: startPage - 1 + progress.requestedPages,
          movies: movieCatalog.length + progress.movies,
        }),
      })
      setMovieCatalog((current) => {
        const merged = mergeCatalog(current, movies)
        void saveCachedCatalog(merged)
        return merged
      })
      setCatalogSource('live')
      setFeedbackNotice(`${movies.length} more live movies added to the ranking pool`)
    } catch {
      setFeedbackNotice('Could not load more movies right now')
    } finally {
      setCatalogLoading(false)
    }
  }

  const searchCatalog = async () => {
    const apiUrl = import.meta.env.VITE_MOVIE_API_URL as string | undefined
    if (!apiUrl || catalogQuery.trim().length < 2 || catalogSearching) return
    setCatalogSearching(true)
    try {
      const movies = await searchMovieCatalog(apiUrl, catalogQuery)
      setMovieCatalog((current) => {
        const merged = mergeCatalog(current, movies)
        void saveCachedCatalog(merged)
        return merged
      })
      setActiveGenres([])
      setActivePlatforms([])
      setActiveMoods([])
      setMaxRuntime(undefined)
      if (movies[0]) setSelectedId(movies[0].id)
      setFeedbackNotice(movies.length ? `${movies.length} search results added; showing ${movies[0].title}` : 'No matching movie found')
    } catch {
      setFeedbackNotice('Movie search is temporarily unavailable')
    } finally {
      setCatalogSearching(false)
    }
  }

  const trainActiveNeuralModel = async () => {
    if (!activeProfile) return
    setNeuralTraining(true)
    setFeedbackNotice(`Training neural model for ${activeProfile.name} locally...`)
    try {
      const result = await trainNeuralTaste(movieCatalog, activeNeuralExamples)
      setNeuralResults((current) => ({ ...current, [activeProfile.id]: result }))
      if (result.status === 'ready') {
        const modeled = scoreCandidates(movieCatalog, profiles, { maxRuntime, genres: activeGenres, platforms: activePlatforms, moods: activeMoods, rankingMode, neuralScores: { [activeProfile.id]: result.predictions }, neuralBlend: result.confidence / 200 })
        const comparison = compareRankings(baselineRecommendations, modeled)
        const experiment: ExperimentRecord = { id: crypto.randomUUID(), timestamp: Date.now(), profileId: activeProfile.id, model: 'MLP', examples: result.examples ?? 0, confidence: result.confidence, loss: result.loss ?? 0, ...comparison }
        setExperiments((current) => [experiment, ...current].slice(0, 12))
        setActiveModel('mlp')
        setFeedbackNotice(`Neural model ready · ${result.confidence}% confidence from ${result.examples} examples`)
        setGraphMode('neural')
      } else {
        setFeedbackNotice(`Neural model needs ${Math.max(0, 4 - activeNeuralExamples.length)} more labeled films`)
      }
    } catch {
      setFeedbackNotice('Neural training failed; explainable scoring remains active')
    } finally {
      setNeuralTraining(false)
    }
  }

  const trainActiveGraphModel = async () => {
    if (!activeProfile) return
    setNeuralTraining(true)
    setFeedbackNotice(`Training graph convolutional network for ${activeProfile.name}...`)
    try {
      const result = await trainGraphNeuralTaste(movieCatalog, activeNeuralExamples)
      setGraphNeuralResults((current) => ({ ...current, [activeProfile.id]: result }))
      if (result.status === 'ready') {
        const modeled = scoreCandidates(movieCatalog, profiles, { maxRuntime, genres: activeGenres, platforms: activePlatforms, moods: activeMoods, rankingMode, neuralScores: { [activeProfile.id]: result.predictions }, neuralBlend: result.confidence / 200 })
        const comparison = compareRankings(baselineRecommendations, modeled)
        const experiment: ExperimentRecord = { id: crypto.randomUUID(), timestamp: Date.now(), profileId: activeProfile.id, model: 'GCN', examples: result.examples ?? 0, confidence: result.confidence, loss: result.loss ?? 0, ...comparison }
        setExperiments((current) => [experiment, ...current].slice(0, 12))
        setActiveModel('gcn')
        setGraphMode('gnn')
        setFeedbackNotice(`GCN ready · ${result.confidence}% confidence · ${result.graphStats?.edges} graph edges`)
      } else {
        setFeedbackNotice(`GCN needs ${Math.max(0, 4 - activeNeuralExamples.length)} more labeled films`)
      }
    } catch {
      setFeedbackNotice('GCN training failed; explainable scoring remains active')
    } finally {
      setNeuralTraining(false)
    }
  }

  const runAgentMission = async () => {
    if (!activeProfile || agentStatus === 'running') return
    setAgentStatus('running')
    const parsed = parseConstraintPrompt(agentGoal)
    const effectiveRankingMode = parsed.rankingMode ?? rankingMode
    const scopedRecommendations = scoreCandidates(movieCatalog, profiles, {
      maxRuntime: parsed.maxRuntime,
      genres: parsed.genres,
      platforms: parsed.platforms,
      moods: parsed.moods,
      rankingMode: effectiveRankingMode,
    })
    const top = scopedRecommendations[0]
    let trainedResult: GraphNeuralResult | undefined

    const result = await runDecisionAgent({
      goal: agentGoal,
      constraintsApplied: false,
      candidateCount: scopedRecommendations.length,
      labeledExamples: activeNeuralExamples.length,
      modelStatus: activeGraphNeuralResult?.status === 'ready' ? 'ready' : 'untrained',
      modelsCompared: experiments.some(({ profileId, model }) => profileId === activeProfile.id && model === 'GCN'),
      disagreement: top ? getDisagreement(top) : 0,
      topMovieId: top?.movie.id,
      topMovieTitle: top?.movie.title,
    }, {
      parseConstraints: async () => {
        setConstraintPrompt(agentGoal)
        setActiveGenres(parsed.genres ?? [])
        setActivePlatforms(parsed.platforms ?? [])
        setActiveMoods(parsed.moods ?? [])
        setMaxRuntime(parsed.maxRuntime)
        setRankingMode(effectiveRankingMode)
      },
      trainGcn: async () => {
        trainedResult = await trainGraphNeuralTaste(movieCatalog, activeNeuralExamples)
        setGraphNeuralResults((current) => ({ ...current, [activeProfile.id]: trainedResult! }))
        if (trainedResult.status === 'ready') {
          setActiveModel('gcn')
          setGraphMode('gnn')
        }
      },
      compareModels: async () => {
        const predictionSource = trainedResult?.status === 'ready' ? trainedResult : activeGraphNeuralResult
        if (!predictionSource || predictionSource.status !== 'ready') return
        const modeled = scoreCandidates(movieCatalog, profiles, {
          maxRuntime: parsed.maxRuntime,
          genres: parsed.genres,
          platforms: parsed.platforms,
          moods: parsed.moods,
          rankingMode: effectiveRankingMode,
          neuralScores: { [activeProfile.id]: predictionSource.predictions },
          neuralBlend: predictionSource.confidence / 200,
        })
        const comparison = compareRankings(scopedRecommendations, modeled)
        const experiment: ExperimentRecord = { id: crypto.randomUUID(), timestamp: Date.now(), profileId: activeProfile.id, model: 'GCN', examples: predictionSource.examples ?? 0, confidence: predictionSource.confidence, loss: predictionSource.loss ?? 0, ...comparison }
        setExperiments((current) => [experiment, ...current].slice(0, 12))
      },
    })

    setAgentTrace((current) => [...result.trace, ...current].slice(0, 30))
    setAgentStatus(result.status === 'awaiting-input' ? 'awaiting-input' : result.status === 'awaiting-approval' ? 'awaiting-approval' : 'idle')
    if (top) setSelectedId(top.movie.id)
    const waitingAction = result.trace.at(-1)?.tool
    if (waitingAction === 'ask_preference') {
      setTasteQuestion(chooseTasteQuestion(scopedRecommendations.map(({ movie }) => movie), activeProfile, new Set(activeNeuralExamples.map(({ movieId }) => movieId))) ?? null)
      setFeedbackNotice('Agent paused for one informative preference')
    } else if (waitingAction === 'request_relaxation') {
      setFeedbackNotice('Agent requests approval to relax constraints')
    } else if (waitingAction === 'propose_pick') {
      setFeedbackNotice(`Agent proposes ${top?.movie.title ?? 'the top candidate'} for approval`)
    }
  }

  const resolveAgentApproval = (approved: boolean) => {
    const waiting = agentTrace.find(({ status }) => status === 'waiting')
    if (!waiting) return
    setAgentTrace((current) => current.map((entry) => entry.id === waiting.id ? { ...entry, status: 'completed', observation: `${entry.observation} Human ${approved ? 'approved' : 'rejected'}.` } : entry))
    if (approved && waiting.tool === 'request_relaxation') {
      setActiveGenres([])
      setActivePlatforms([])
      setActiveMoods([])
      setMaxRuntime(undefined)
      setFeedbackNotice('Constraints relaxed with human approval')
    } else if (approved && waiting.tool === 'propose_pick' && selected && activeProfile) {
      setProfiles((current) => current.map((profile) => profile.id === activeProfile.id ? applyTasteFeedback(profile, selected.movie, 'pick') : profile))
      setFeedbackEvents((current) => [...current, { id: crypto.randomUUID(), profileId: activeProfile.id, movieId: selected.movie.id, action: 'pick', timestamp: Date.now() }])
      setFeedbackNotice(`${selected.movie.title} approved as tonight's pick`)
    } else if (!approved) {
      pickAnother()
      setFeedbackNotice('Agent proposal rejected; moved to the next candidate')
    }
    setAgentStatus('idle')
  }

  const applyConstraintPrompt = () => {
    const parsed = parseConstraintPrompt(constraintPrompt)
    setActiveGenres(parsed.genres ?? [])
    setActivePlatforms(parsed.platforms ?? [])
    setActiveMoods(parsed.moods ?? [])
    setMaxRuntime(parsed.maxRuntime)
    if (parsed.rankingMode) setRankingMode(parsed.rankingMode)
    setPromptInterpretation(describeConstraints(parsed))
    setFeedbackNotice(constraintPrompt.trim() ? 'Agent translated your request into filters' : 'Agent filters cleared')
  }

  const createSnapshot = (): SessionSnapshot => ({
    profiles,
    maxRuntime,
    activeGenres,
    activePlatforms,
    activeMoods,
    rankingMode,
    activeProfileId,
    nodePositions: Object.fromEntries(renderedNodes.map(({ id, position }) => [id, position])),
    feedbackEvents,
    neuralResults,
    graphNeuralResults,
    experiments,
    agentTrace,
  })

  const shareRoom = async () => {
    const room = encodeSharedRoom(createSnapshot())
    const url = `${window.location.origin}${window.location.pathname}?room=${room}`
    try {
      await navigator.clipboard.writeText(url)
      setFeedbackNotice('Private room link copied; raw history was excluded')
    } catch {
      setFeedbackNotice('Browser blocked clipboard access')
    }
  }

  const exportPortfolioReport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      catalog: { source: catalogSource, movies: movieCatalog.length, region: 'ES' },
      configuration: { maxRuntime, activeGenres, activePlatforms, activeMoods, rankingMode, activeModel, neuralEnabled, neuralBlend },
      learning: metrics,
      models: { mlp: neuralResults, gcn: graphNeuralResults },
      experiments,
      agent: { evaluation: agentEvaluation, latestRunScore: latestAgentScore, trace: agentTrace },
      privacy: 'Raw ratings and watched-film history intentionally excluded.',
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `reel-circle-report-${new Date().toISOString().slice(0, 10)}.json`
    anchor.click()
    URL.revokeObjectURL(url)
    setFeedbackNotice('Portfolio experiment report downloaded')
  }

  const downloadJson = (filename: string, value: unknown) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const exportUserData = () => {
    downloadJson(`reel-circle-backup-${new Date().toISOString().slice(0, 10)}.json`, createSnapshot())
    setFeedbackNotice(locale === 'es' ? 'Copia de seguridad descargada' : 'Local data backup downloaded')
  }

  const deleteUserData = async () => {
    if (!window.confirm(locale === 'es' ? '¿Eliminar todos los perfiles, valoraciones y modelos locales?' : 'Delete all local profiles, ratings, and models?')) return
    await clearAllLocalData()
    localStorage.removeItem('reel-circle-onboarding')
    setShowOnboarding(true)
    setRestoredPositions({})
    setProfiles(demoProfiles)
    setMaxRuntime(120)
    setActiveGenres([])
    setActivePlatforms([])
    setActiveMoods([])
    setRankingMode('balanced')
    setActiveProfileId(demoProfiles[0].id)
    setFeedbackEvents([])
    setNeuralResults({})
    setGraphNeuralResults({})
    setExperiments([])
    setAgentTrace([])
    setAgentStatus('idle')
    setConstraintPrompt('')
    setFeedbackNotice(locale === 'es' ? 'Datos locales eliminados' : 'Local data deleted')
    window.history.replaceState({}, '', window.location.pathname)
  }

  const removeProfile = (profileId: string) => {
    setProfiles((current) => current.filter(({ id }) => id !== profileId))
    if (activeProfileId === profileId) {
      setActiveProfileId(profiles.find(({ id }) => id !== profileId)?.id ?? '')
    }
  }

  const recordFeedback = (feedback: TasteFeedback) => {
    if (!selected || !activeProfile) return
    setProfiles((current) => current.map((profile) => profile.id === activeProfile.id ? applyTasteFeedback(profile, selected.movie, feedback) : profile))
    setFeedbackEvents((current) => [...current, { id: crypto.randomUUID(), profileId: activeProfile.id, movieId: selected.movie.id, action: feedback, timestamp: Date.now() }])
    setNeuralResults((current) => Object.fromEntries(Object.entries(current).filter(([profileId]) => profileId !== activeProfile.id)))
    setGraphNeuralResults((current) => Object.fromEntries(Object.entries(current).filter(([profileId]) => profileId !== activeProfile.id)))
    const labels: Record<TasteFeedback, string> = { skip: 'Skipped', pick: 'Picked', love: 'Loved and added to watched' }
    setFeedbackNotice(`${activeProfile.name}: ${labels[feedback]} ${selected.movie.title}`)
    if (feedback === 'love') setSelectedId('')
  }

  const openTasteQuestion = () => {
    if (!activeProfile) return
    comparisonOpener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    setTasteQuestion(chooseTasteQuestion(recommendations.map(({ movie }) => movie), activeProfile, new Set(activeNeuralExamples.map(({ movieId }) => movieId))) ?? null)
  }

  const answerTasteQuestion = (chosen: Movie) => {
    if (!activeProfile || !tasteQuestion) return
    const rejected = tasteQuestion.find(({ id }) => id !== chosen.id)
    if (!rejected) return
    setProfiles((current) => current.map((profile) => profile.id === activeProfile.id ? applyComparisonFeedback(profile, chosen, rejected) : profile))
    setFeedbackEvents((current) => [...current, { id: crypto.randomUUID(), profileId: activeProfile.id, movieId: chosen.id, rejectedMovieId: rejected.id, action: 'comparison', timestamp: Date.now() }])
    setNeuralResults((current) => Object.fromEntries(Object.entries(current).filter(([profileId]) => profileId !== activeProfile.id)))
    setGraphNeuralResults((current) => Object.fromEntries(Object.entries(current).filter(([profileId]) => profileId !== activeProfile.id)))
    setFeedbackNotice(`${activeProfile.name} prefers ${chosen.title} over ${rejected.title}`)
    if (agentStatus === 'awaiting-input') {
      setAgentTrace((current) => current.map((entry) => entry.status === 'waiting' ? { ...entry, status: 'completed', observation: `${entry.observation} User selected ${chosen.title}.` } : entry))
      setAgentStatus('idle')
    }
    setTasteQuestion(null)
  }

  const handleGraphNodeClick = (nodeId: string) => {
    if (profiles.some(({ id }) => id === nodeId)) {
      setActiveProfileId(nodeId)
    } else if (nodeId.startsWith('genre-')) {
      toggleGenre(nodeId.replace('genre-', ''))
    } else if (recommendations.some(({ movie }) => movie.id === nodeId)) {
      setSelectedId(nodeId)
    }
  }

  const pickAnother = () => {
    if (recommendations.length < 2) return
    const currentIndex = recommendations.findIndex(({ movie }) => movie.id === selected?.movie.id)
    setSelectedId(recommendations[(currentIndex + 1) % Math.min(3, recommendations.length)].movie.id)
  }

  const toggleLabMode = () => {
    setLabMode((current) => {
      if (current) setGraphMode('taste')
      return !current
    })
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Reel Circle home"><span className="brand-mark"><i /><i /><i /></span>REEL CIRCLE</a>
        <div className="topbar-meta"><div className="privacy-note"><span /> {hydrated ? t('savedLocally') : t('loadingSession')}</div><button type="button" className="header-icon" onClick={() => void shareRoom()} title={t('shareRoom')} aria-label={t('shareRoom')}><Share2 size={16} /></button><button type="button" className="locale-toggle" onClick={() => setLocale((current) => current === 'en' ? 'es' : 'en')} aria-label={locale === 'en' ? 'Cambiar a español' : 'Switch to English'}>{locale === 'en' ? 'ES' : 'EN'}</button><button type="button" className="header-icon" onClick={() => setThemePreference((current) => resolveTheme(current, window.matchMedia('(prefers-color-scheme: dark)').matches) === 'dark' ? 'light' : 'dark')} title={resolveTheme(themePreference, window.matchMedia('(prefers-color-scheme: dark)').matches) === 'dark' ? t('switchLight') : t('switchDark')} aria-label={resolveTheme(themePreference, window.matchMedia('(prefers-color-scheme: dark)').matches) === 'dark' ? t('switchLight') : t('switchDark')}>{resolveTheme(themePreference, window.matchMedia('(prefers-color-scheme: dark)').matches) === 'dark' ? <Sun size={16} /> : <Moon size={16} />}</button><button type="button" className={labMode ? 'lab-toggle active' : 'lab-toggle'} onClick={toggleLabMode}><FlaskConical size={15} /> {labMode ? t('closeLab') : t('openLab')}</button></div>
      </header>

      <section className="agent-console primary-agent" aria-label="Conversational movie agent">
        <div className="agent-command">
          <MessageSquareText size={19} />
          <label htmlFor="constraint-prompt"><strong>{t('promptTitle')}</strong><small>{t('promptHelp')}</small></label>
          <input id="constraint-prompt" value={constraintPrompt} onChange={(event) => setConstraintPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') applyConstraintPrompt() }} placeholder={t('promptPlaceholder')} />
          <button type="button" onClick={applyConstraintPrompt}><Sparkles size={15} /> {t('apply')}</button>
        </div>
        {promptInterpretation && <div className="agent-interpretation"><span><CheckCircle2 size={14} /> {t('understood')}</span><div>{promptInterpretation.map((constraint) => <em key={constraint}>{displayValue(locale, constraint)}</em>)}</div><small>{recommendations.length} {t('matchesNow')}</small></div>}
        {labMode && <div className="agent-settings">
          <div><label><Target size={14} /> {locale === 'es' ? 'Estrategia del grupo' : 'Group strategy'}</label><div className="strategy-options">{([
            ['balanced', locale === 'es' ? 'Equilibrada' : 'Balanced'], ['average', locale === 'es' ? 'Media' : 'Average'], ['least-misery', locale === 'es' ? 'Que nadie pierda' : 'No one loses'], ['nash', locale === 'es' ? 'Nash justa' : 'Nash fair'],
          ] as Array<[RankingMode, string]>).map(([mode, label]) => <button type="button" key={mode} className={rankingMode === mode ? 'active' : ''} onClick={() => setRankingMode(mode)}>{label}</button>)}</div></div>
          <div><label><Sparkles size={14} /> {locale === 'es' ? 'Ambiente' : 'Mood'}</label><div className="mood-options">{moods.map((mood) => <button type="button" key={mood} className={activeMoods.includes(mood) ? 'active' : ''} onClick={() => toggleMood(mood)}>{displayValue(locale, mood)}</button>)}</div></div>
          <div className="neural-control"><label><Cpu size={14} /> {locale === 'es' ? 'Laboratorio de modelos' : 'Model lab'} · {activeProfile?.name ?? (locale === 'es' ? 'Sin perfil' : 'No profile')}</label><div><span className={`neural-status ${activeModelResult?.status === 'ready' ? 'ready' : ''}`}>{activeModelResult?.status === 'ready' ? `${activeModel.toUpperCase()} · ${activeModelResult.confidence}% ${locale === 'es' ? 'confianza' : 'confidence'}` : `${activeNeuralExamples.length}/4 ${locale === 'es' ? 'etiquetas' : 'labels'}`}</span><button type="button" onClick={() => void trainActiveNeuralModel()} disabled={neuralTraining || !activeProfile}>{neuralTraining ? <Activity size={14} /> : <Cpu size={14} />}{neuralTraining ? (locale === 'es' ? 'Entrenando...' : 'Training...') : (locale === 'es' ? 'Entrenar MLP' : 'Train MLP')}</button><button type="button" onClick={() => void trainActiveGraphModel()} disabled={neuralTraining || !activeProfile}>{neuralTraining ? <Activity size={14} /> : <Network size={14} />}{locale === 'es' ? 'Entrenar GCN' : 'Train GCN'}</button><button type="button" className={neuralEnabled ? 'active' : ''} onClick={() => setNeuralEnabled((current) => !current)} disabled={activeModelResult?.status !== 'ready'}>{neuralEnabled ? `${locale === 'es' ? 'Mezcla' : 'Blend'} ${Math.round(neuralBlend * 100)}%` : (locale === 'es' ? 'Mezcla desactivada' : 'Blend off')}</button></div></div>
        </div>}
      </section>

      <section className="intro" id="top">
        <div className="intro-main"><p className="eyebrow"><Sparkles size={15} /> {t('decisionRoom')}</p><h1>{t('heroTitle').split('\n').map((line, index) => <span key={line}>{line}{index === 0 && <br />}</span>)}</h1><p className="intro-copy">{t('heroCopy')}</p><div className="intro-actions"><button type="button" onClick={openTasteQuestion} disabled={recommendations.length < 2}><GitCompareArrows size={16} /> {t('compareMovies')}</button><button type="button" onClick={() => document.querySelector('#decision')?.scrollIntoView({ behavior: 'smooth' })}><Target size={16} /> {t('viewShortlist')}</button></div></div>
        <div className={`session-summary ${labMode ? 'lab-open' : ''}`}>
          <div><Users size={17} /><strong>{profiles.length}</strong><small>{t('viewers')}</small></div>
          <div><Film size={17} /><strong>{catalogLoading ? catalogProgress.movies : recommendations.length}</strong><small>{catalogLoading ? `${locale === 'es' ? 'Cargando página' : 'Loading page'} ${catalogProgress.loadedPages}/${catalogProgress.requestedPages}` : `${movieCatalog.length} ${t('films')}`}</small></div>
          <div><Target size={17} /><strong>{activeProfile?.name ?? '—'}</strong><small>{t('learning')}</small></div>
          {labMode && <div><BarChart3 size={17} /><strong>{metrics.acceptanceRate}%</strong><small>{t('accepted')}</small></div>}
        </div>
      </section>

      <nav className="workflow-nav" aria-label="Movie night workflow">
        <a href="#people"><span>01</span><strong>{t('addPeople')}</strong><small>{t('importTaste')}</small></a>
        <a href="#filters"><span>02</span><strong>{t('setLimits')}</strong><small>{t('narrowField')}</small></a>
        <a href="#explore"><span>03</span><strong>{t('explore')}</strong><small>{t('tasteMap')}</small></a>
        <a href="#decision"><span>04</span><strong>{t('decide')}</strong><small>{t('finalShortlist')}</small></a>
      </nav>

      {showOnboarding && <section className="onboarding" aria-label={locale === 'es' ? 'Primeros pasos' : 'Getting started'}>
        <div><Sparkles size={19} /><span><strong>{locale === 'es' ? 'Empieza en tres pasos' : 'Start in three steps'}</strong><small>{locale === 'es' ? 'Tus datos se procesan y guardan en este navegador.' : 'Your data is processed and stored in this browser.'}</small></span></div>
        <ol><li><span>1</span>{locale === 'es' ? 'Importa tu ZIP de Letterboxd' : 'Import your Letterboxd ZIP'}</li><li><span>2</span>{locale === 'es' ? 'Elige plataformas y límites' : 'Choose platforms and limits'}</li><li><span>3</span>{locale === 'es' ? 'Compara o elige una recomendación' : 'Compare or pick a recommendation'}</li></ol>
        <button type="button" onClick={() => { localStorage.setItem('reel-circle-onboarding', 'done'); setShowOnboarding(false) }}><X size={15} /><span>{locale === 'es' ? 'Cerrar' : 'Dismiss'}</span></button>
      </section>}

      <section className="setup-grid">
      <section className="people-strip" id="people" aria-label="People in this session">
        <div className="section-label"><span>01</span><div><strong>{t('whosWatching')}</strong><small>{profiles.length} {profiles.length === 1 ? t('personConnected') : t('peopleConnected')}</small></div></div>
        <div className="people-list">
          {profiles.map((profile) => <div className={`person-chip ${profile.id === activeProfile?.id ? 'active' : ''}`} key={profile.id} style={{ '--person': profile.color } as CSSProperties}>
            <button className="person-select" type="button" onClick={() => setActiveProfileId(profile.id)}><span className="avatar">{profile.name.slice(0, 1).toUpperCase()}</span><span><strong>{profile.name}</strong><small>{profile.id === activeProfile?.id ? t('learningNow') : `${profile.ratings.length} ${t('ratings')}`}</small></span></button>
            <button className="person-remove" type="button" aria-label={`${t('remove')} ${profile.name}`} title={`${t('remove')} ${profile.name}`} onClick={() => removeProfile(profile.id)}><X size={14} /></button>
          </div>)}
          <button className="add-person" type="button" onClick={() => fileInput.current?.click()}><Plus size={18} /> {t('addLetterboxd')}</button>
          <input ref={fileInput} type="file" hidden onChange={(event) => void importProfile(event.target.files?.[0])} />
        </div>
      </section>

      <section className="control-strip" id="filters">
        <div className="section-label"><span>02</span><div><strong>{t('boundaries')}</strong><small>{t('boundariesHelp')}</small></div></div>
        <div className="filter-block"><label><Clock3 size={16} /> {t('runtime')}</label><div className="segments">
          {[{ label: t('any'), value: undefined }, { label: t('under90'), value: 90 }, { label: t('under2h'), value: 120 }].map((option) => <button type="button" key={option.label} className={maxRuntime === option.value ? 'active' : ''} onClick={() => setMaxRuntime(option.value)}>{option.label}</button>)}
        </div></div>
        <div className="filter-block"><label><MonitorPlay size={16} /> {t('platformSpain')}</label><div className="platform-options">{platforms.map((platform) => <button type="button" key={platform} className={activePlatforms.includes(platform) ? 'active' : ''} onClick={() => togglePlatform(platform)}>{platform}</button>)}</div></div>
        <div className="filter-block"><label><Film size={16} /> {t('genre')}</label><div className="genre-options">{genres.map((genre) => <button type="button" key={genre} className={activeGenres.includes(genre) ? 'active' : ''} onClick={() => toggleGenre(genre)}>{displayValue(locale, genre)}</button>)}</div></div>
        <div className="catalog-browser"><label htmlFor="catalog-search"><Database size={15} /> {t('liveCatalog')} · {movieCatalog.length} {t('films')}</label><div><input id="catalog-search" value={catalogQuery} onChange={(event) => setCatalogQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void searchCatalog() }} placeholder={t('searchTitle')} /><button type="button" onClick={() => void searchCatalog()} disabled={catalogSearching || catalogQuery.trim().length < 2}>{catalogSearching ? <LoaderCircle size={14} /> : <Search size={14} />} {t('search')}</button><button type="button" onClick={() => void loadMoreCatalog()} disabled={catalogLoading}>{catalogLoading ? <LoaderCircle size={14} /> : <Plus size={14} />} {t('loadMore')}</button></div></div>
        {catalogError && <div className="catalog-status" role="status">{locale === 'es' ? catalogError.replace('Live refresh failed; using cached movies', 'La actualización falló; usando películas guardadas').replace('Live catalog unavailable; using curated movies', 'Catálogo en vivo no disponible; usando selección local') : catalogError}</div>}
      </section>
      </section>

      <section className="workspace" id="explore">
        <div className="graph-panel">
          <div className="panel-heading"><div><p className="eyebrow">{graphMode === 'neural' ? <Cpu size={14} /> : <Network size={14} />} 03 · {graphMode === 'taste' ? t('tasteMap') : graphMode === 'neural' ? 'MLP inspector' : 'Graph neural network'}</p><h2>{graphMode === 'taste' ? t('whereTastesMeet') : graphMode === 'neural' ? (locale === 'es' ? 'Observa cómo una película activa las neuronas' : 'Watch a movie activate neurons') : (locale === 'es' ? 'Inspecciona el paso de mensajes' : 'Inspect feature message passing')}</h2></div><div className="graph-actions">{labMode && <div className="graph-mode"><button type="button" className={graphMode === 'taste' ? 'active' : ''} onClick={() => setGraphMode('taste')}><Network size={14} /><span>{locale === 'es' ? 'Gustos' : 'Taste'}</span></button><button type="button" className={graphMode === 'neural' ? 'active' : ''} onClick={() => setGraphMode('neural')}><Cpu size={14} /><span>MLP</span></button><button type="button" className={graphMode === 'gnn' ? 'active' : ''} onClick={() => setGraphMode('gnn')}><Network size={14} /><span>GCN</span></button></div>}{labMode && graphMode === 'taste' && <button type="button" className={showDeepSignals ? 'active' : ''} title={locale === 'es' ? 'Alternar señales aprendidas' : 'Toggle deep learned signals'} onClick={() => setShowDeepSignals((current) => !current)}><Layers3 size={14} /><span>{locale === 'es' ? 'Señales' : 'Deep'}</span></button>}<span className="live-badge">{graphMode === 'neural' ? '34 · 12 · 6 · 1' : graphMode === 'gnn' ? '2 GCN LAYERS' : t('liveGraph')}</span>{graphMode === 'taste' && <button type="button" title={t('resetGraph')} aria-label={t('resetGraph')} onClick={() => { setRestoredPositions({}); setRenderedNodes(graphNodes) }}><RotateCcw size={14} /></button>}</div></div>
          <div className="graph-canvas">{graphMode === 'taste' ? <><div className={`graph-column-labels ${showDeepSignals ? 'deep' : ''}`}><span>{t('people')}</span><span>{t('genres')}</span>{showDeepSignals && <span>{t('signals')}</span>}<span>{t('films')}</span></div>
            <ReactFlow nodes={renderedNodes} edges={graphEdges} onNodesChange={onNodesChange} onNodeDragStop={(_, node) => setRestoredPositions((current) => ({ ...current, [node.id]: node.position }))} fitView nodesDraggable nodesConnectable={false} onNodeClick={(_, node) => handleGraphNodeClick(node.id)}><Background color="#d8d3c8" gap={24} size={1} /><Controls showInteractive={false} /></ReactFlow></> : graphMode === 'neural' ? <><div className="graph-column-labels neural-labels"><span>34 inputs</span><span>12 ReLU</span><span>6 ReLU</span><span>Sigmoid</span></div><ReactFlow nodes={neuralGraphNodes} edges={neuralGraphEdges} fitView nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}><Background color="#c6d7d0" gap={22} size={1} /><Controls showInteractive={false} /></ReactFlow><div className="neural-caption">Select a film to inspect activations · {activeNeuralExamples.length} labels · {activeNeuralResult?.status === 'ready' ? `loss ${activeNeuralResult.loss?.toFixed(3)}` : 'waiting for MLP'}</div></> : <><div className="graph-column-labels gcn-labels"><span>Feature nodes</span><span>Normalized messages</span><span>Movie embeddings</span></div><ReactFlow nodes={graphNeuralNodes} edges={graphNeuralEdges} fitView nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}><Background color="#c7d8df" gap={22} size={1} /><Controls showInteractive={false} /></ReactFlow><div className="neural-caption">GCN: H² = ReLU(Â ReLU(Â X W⁰) W¹) · {activeGraphNeuralResult?.graphStats?.edges ?? 0} edges · {activeGraphNeuralResult?.status === 'ready' ? `loss ${activeGraphNeuralResult.loss?.toFixed(3)}` : 'waiting for GCN'}</div></>}
          </div>
        </div>

        <aside className="results-panel" id="decision">
          <div className="panel-heading"><div><p className="eyebrow"><Sparkles size={14} /> 04 · {t('decide')}</p><h2>{t('tonightShortlist')}</h2></div><span>{recommendations.length} {t('matches')}</span></div>
          {selected ? <div className="hero-pick">
            <div className="poster-wrap"><img src={selected.movie.poster} alt={`${selected.movie.title} poster`} onError={usePosterFallback} /><span>{selected.score}%</span></div>
            <div className="pick-copy"><p className="rank">{t('bestBet')}</p><h3>{selected.movie.title}</h3><p className="metadata">{selected.movie.year} · {selected.movie.runtime} min · {displayValue(locale, selected.movie.tone ?? '')}</p><div className="availability">{selected.movie.platforms.map((platform) => <span key={platform}>{platform}</span>)}</div><p className="reason">{localizeReason(locale, selected.reasons[0])}. {localizeReason(locale, selected.reasons[1])}.</p>
              <div className="profile-scores">{selected.matchByProfile.map((match) => { const profile = profiles.find(({ id }) => id === match.profileId); return <span key={match.profileId} style={{ '--person': profile?.color } as CSSProperties}><i />{profile?.name} {match.score}%</span> })}</div>
            </div>
          </div> : <div className="empty-state"><Film size={30} /><strong>{t('noMatches')}</strong><span>{t('widenFilters')}</span></div>}
          {selected && activeProfile && <div className="taste-agent">
            <div className="agent-heading"><BrainCircuit size={17} /><span><strong>{t('tasteLearner')} · {activeProfile.name}</strong><small>{feedbackNotice || t('readyFeedback')}</small></span></div>
            <div className="feedback-actions">
              <button type="button" onClick={() => recordFeedback('skip')} title={t('skip')}><ThumbsDown size={15} /> {t('skip')}</button>
              <button type="button" onClick={() => recordFeedback('pick')} title={t('picked')}><Check size={15} /> {t('picked')}</button>
              <button type="button" onClick={() => recordFeedback('love')} title={t('loved')}><Heart size={15} /> {t('loved')}</button>
            </div>
            <button className="comparison-trigger" type="button" onClick={openTasteQuestion} disabled={recommendations.length < 2}><GitCompareArrows size={15} /> {t('compareMovies')}</button>
          </div>}
          <div className="subsection-label"><span>{t('rankedAlternatives')}</span><small>{t('groupScore')}</small></div>
          <div className="runner-ups">{recommendations.slice(0, 4).map((item, index) => <button type="button" key={item.movie.id} className={item.movie.id === selected?.movie.id ? 'selected' : ''} onClick={() => setSelectedId(item.movie.id)}><span className="list-rank">0{index + 1}</span><img src={item.movie.poster} alt="" onError={usePosterFallback} /><span className="list-title"><strong>{item.movie.title}</strong><small>{item.movie.year} · {item.movie.runtime} min</small></span><span className="score">{item.score}</span></button>)}</div>
          <button className="shuffle-button" type="button" onClick={pickAnother} disabled={recommendations.length < 2}><Shuffle size={17} /> {t('pickAnother')}</button>
        </aside>
      </section>

      <section className="summary-row" aria-label="Decision summary">
        <article><span className="summary-icon"><Users size={18} /></span><div><small>{t('sharedTaste')}</small><strong>{selected ? localizeReason(locale, selected.reasons[0]) : t('addPreferences')}</strong><p>{profiles.length} {profiles.length === 1 ? t('profileContributes') : t('profilesContribute')}</p></div></article>
        <article><span className="summary-icon orange"><Target size={18} /></span><div><small>{t('topResult')}</small><strong>{selected?.movie.title ?? t('noMatchYet')}</strong><p>{selected ? `${selected.score}% · ${selected.movie.runtime} min` : t('widenCurrent')}</p></div></article>
        <article><span className="summary-icon blue"><Thermometer size={18} /></span><div><small>{t('groupBalance')}</small><strong>{selectedDisagreement <= 15 ? t('strongConsensus') : selectedDisagreement <= 30 ? t('goodCompromise') : t('mixedOpinions')}</strong><p>{selectedDisagreement} {t('pointSpread')}</p></div></article>
      </section>

      {labMode && <section className="lab-accordions" aria-label="AI lab details">
        <details>
          <summary><span><FlaskConical size={17} /><strong>{t('experimentLab')}</strong><small>{t('experimentHelp')}</small></span><ChevronRight size={17} /></summary>
          <div className="experiment-lab" aria-label="Model experiment results">
            <div className="lab-heading"><div><p className="eyebrow"><FlaskConical size={14} /> {t('experimentLab')} · {catalogSource === 'live' ? (locale === 'es' ? 'catálogo en vivo' : 'live catalog') : catalogSource === 'cached' ? (locale === 'es' ? 'catálogo guardado' : 'cached catalog') : (locale === 'es' ? 'catálogo local' : 'curated catalog')}</p><h2>{locale === 'es' ? 'Compara modelos e inspecciona decisiones' : 'Compare models, inspect decisions'}</h2></div><div className="lab-tools"><div className="model-switch"><button type="button" className={activeModel === 'mlp' ? 'active' : ''} onClick={() => setActiveModel('mlp')} disabled={activeNeuralResult?.status !== 'ready'}>MLP</button><button type="button" className={activeModel === 'gcn' ? 'active' : ''} onClick={() => setActiveModel('gcn')} disabled={activeGraphNeuralResult?.status !== 'ready'}>GCN</button></div><button type="button" className="reset-session" onClick={exportUserData}><Download size={14} /> {locale === 'es' ? 'Copia' : 'Backup'}</button><button type="button" className="reset-session danger" onClick={() => void deleteUserData()}><Trash2 size={14} /> {locale === 'es' ? 'Eliminar datos' : 'Delete data'}</button></div></div>
            <div className="lab-metrics">
              <article><GitCompareArrows size={17} /><span><strong>{rankingComparison.topChanged ? 'Changed' : 'Stable'}</strong><small>Top recommendation</small></span></article>
              <article><Layers3 size={17} /><span><strong>{rankingComparison.topThreeOverlap}%</strong><small>Top-3 overlap</small></span></article>
              <article><Activity size={17} /><span><strong>{rankingComparison.meanRankShift}</strong><small>Mean rank shift</small></span></article>
              <article className={selectedDisagreement > 25 ? 'warning' : ''}><Thermometer size={17} /><span><strong>{selectedDisagreement} pts</strong><small>Group disagreement</small></span></article>
            </div>
            <div className="lab-insights">
              <div className="path-card"><Route size={16} /><span><strong>Strongest explanation path</strong><small>{selectedPath ? selectedPath.path.join(' → ') : 'Collect feedback to reveal a path'}</small></span></div>
              <label className="heatmap-toggle"><input type="checkbox" checked={showHeatmap} onChange={(event) => setShowHeatmap(event.target.checked)} /><span>Conflict heatmap</span></label>
              <div className="experiment-history"><strong>{experiments.length} runs</strong><span>{experiments.slice(0, 3).map((experiment) => `${experiment.model} ${experiment.confidence}%`).join(' · ') || 'Train a model to log an experiment'}</span></div>
            </div>
          </div>
        </details>
        <details>
          <summary><span><Bot size={17} /><strong>{t('autonomousAgent')}</strong><small>{t('autonomousHelp')}</small></span><ChevronRight size={17} /></summary>
          <div className="autonomy-lab" aria-label="Autonomous agent mission">
            <div className="autonomy-header"><div><p className="eyebrow"><Bot size={14} /> {t('autonomousAgent')}</p><h2>{locale === 'es' ? 'Dale un objetivo e inspecciona cada paso' : 'Give it a goal, inspect every step'}</h2></div><div className="autonomy-actions"><span className={`agent-run-status ${agentStatus}`}><i />{locale === 'es' ? agentStatus.replace('idle', 'inactivo').replace('running', 'ejecutando').replace('awaiting input', 'esperando respuesta').replace('awaiting approval', 'esperando aprobación') : agentStatus.replace('-', ' ')}</span><button type="button" onClick={exportPortfolioReport}><Download size={14} /> {locale === 'es' ? 'Exportar informe' : 'Export report'}</button></div></div>
            <div className="mission-row"><label htmlFor="agent-goal"><strong>Mission</strong><small>The agent can parse, inspect, ask, train, compare, and propose.</small></label><input id="agent-goal" value={agentGoal} onChange={(event) => setAgentGoal(event.target.value)} placeholder="Choose a fair movie under two hours" /><button type="button" onClick={() => void runAgentMission()} disabled={agentStatus === 'running'}>{agentStatus === 'running' ? <Activity size={16} /> : <Play size={16} />}{agentStatus === 'running' ? 'Running...' : 'Run agent'}</button></div>
            <div className="benchmark-prompts"><span>Benchmarks</span>{['Pick a fair movie under two hours', 'Algo divertido en Filmin de menos de 90 minutos', 'A gentle romantic movie everyone can enjoy'].map((goal) => <button type="button" key={goal} onClick={() => setAgentGoal(goal)}>{goal}</button>)}</div>
            <div className="agent-trace"><div className="trace-heading"><span>Execution trace</span><small>{agentTrace.length} tool events · max 8 steps/run</small></div>{agentTrace.length === 0 ? <div className="trace-empty">Run a mission to see planning, tools, observations, and approval gates.</div> : agentTrace.slice(0, 8).map((entry) => <article className={entry.status} key={entry.id}><span className="trace-step">{String(entry.step).padStart(2, '0')}</span><span className="trace-icon">{entry.status === 'completed' ? <CheckCircle2 size={15} /> : <ShieldCheck size={15} />}</span><span className="trace-copy"><strong>{entry.tool.replaceAll('_', ' ')}</strong><small>{entry.rationale}</small><em>{entry.observation}</em></span><span className="trace-state">{entry.status}</span></article>)}</div>
            <div className="agent-eval-grid">
              <article><strong>{agentEvaluation.runs}</strong><small>Runs</small></article>
              <article><strong>{agentEvaluation.averageSteps}</strong><small>Avg. steps</small></article>
              <article><strong>{agentEvaluation.completedStepRate}%</strong><small>Tool success</small></article>
              <article><strong>{agentEvaluation.interventionRate}%</strong><small>Human-in-loop</small></article>
              <article><strong>{agentEvaluation.approvalRate}%</strong><small>Approval rate</small></article>
              <article><strong>{latestAgentScore}</strong><small>Latest score</small></article>
            </div>
            {agentStatus === 'awaiting-approval' && <div className="approval-gate"><ShieldCheck size={18} /><span><strong>Human approval required</strong><small>The agent cannot change hard constraints or commit a pick on its own.</small></span><button type="button" onClick={() => resolveAgentApproval(false)}>Reject</button><button type="button" onClick={() => resolveAgentApproval(true)}>Approve</button></div>}
          </div>
        </details>
      </section>}

      {tasteQuestion && activeProfile && <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) setTasteQuestion(null) }}>
        <section ref={comparisonDialog} className="comparison-dialog" role="dialog" aria-modal="true" aria-labelledby="comparison-title" aria-describedby="comparison-note">
          <header><div><p className="eyebrow"><BrainCircuit size={14} /> {t('activeLearning')} · {activeProfile.name}</p><h2 id="comparison-title">{t('whichRather')}</h2></div><button type="button" aria-label={locale === 'es' ? 'Cerrar comparación' : 'Close taste question'} title={locale === 'es' ? 'Cerrar' : 'Close'} onClick={() => setTasteQuestion(null)} autoFocus><X size={17} /></button></header>
          <div className="comparison-options">
            {tasteQuestion.map((movie) => <button className="comparison-choice" type="button" key={movie.id} onClick={() => answerTasteQuestion(movie)}>
              <img src={movie.poster} alt={`${movie.title} poster`} onError={usePosterFallback} />
              <span><strong>{movie.title}</strong><small>{movie.year} · {movie.runtime} min</small><em>{movie.genres.map((genre) => displayValue(locale, genre)).join(' · ')}</em></span>
            </button>)}
          </div>
          <p className="comparison-note" id="comparison-note">{t('oneChoice')}</p>
        </section>
      </div>}

      <footer><span><Upload size={15} /> {t('importFooter')}</span><span><Database size={14} /> {metrics.interactions} {t('learnedSignals')} · {metrics.comparisons} {t('comparisons')}</span><a href="https://www.themoviedb.org/" target="_blank" rel="noreferrer">TMDB · {locale === 'es' ? 'no avalado ni certificado' : 'not endorsed or certified'}</a><button type="button" onClick={() => void shareRoom()}><Copy size={14} /> {t('copyRoom')}</button></footer>
    </main>
  )
}

export default App