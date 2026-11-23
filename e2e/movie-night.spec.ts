import { expect, test, type Page } from '@playwright/test'

const fullTurn = ['extract_preferences', 'filter_candidates', 'semantic_retrieval', 'rank_candidates', 'validate_results']
const completedTrace = (tools: string[]) => tools.map((tool) => ({ tool, status: 'completed', durationMs: 1, observation: 'ok' }))
const deterministicMeta = { extractionMode: 'deterministic', modelStatus: 'not-configured', durationMs: 5 }

/** Stands in for the agent API: every turn gets what `answer` returns for the request it sent. */
async function answerAgentTurns(page: Page, answer: (request: { decision?: string }) => object) {
  await page.route('**/api/agent', async (route) => {
    const body = JSON.stringify({ ...answer(route.request().postDataJSON()), conversationId: 'conv-e2e' })
    await route.fulfill({ contentType: 'application/json', body })
  })
}

async function askAgent(page: Page, message: string) {
  await page.locator('#constraint-prompt').fill(message)
  await page.getByRole('button', { name: 'Apply', exact: true }).click()
}

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await page.evaluate(async () => {
    localStorage.clear()
    await new Promise<void>((resolve) => {
      const request = indexedDB.deleteDatabase('reel-circle')
      request.onsuccess = () => resolve()
      request.onerror = () => resolve()
      request.onblocked = () => resolve()
    })
  })
  await page.reload()
})

test('completes the primary movie-night workflow', async ({ page }) => {
  await expect(page.getByRole('heading', { name: /Find the film everyone/i })).toBeVisible()
  await page.locator('.locale-toggle').click()
  await expect(page.getByRole('heading', { name: /Encuentra la película/i })).toBeVisible()

  await page.locator('#constraint-prompt').fill('algo divertido de menos de 100 minutos')
  await page.getByRole('button', { name: 'Aplicar', exact: true }).click()
  await expect(page.locator('.agent-interpretation')).toContainText('Comedia')
  await expect(page.locator('.agent-interpretation')).toContainText('Menos de 100 min')

  const compareButton = page.getByRole('button', { name: 'Comparar dos películas' }).first()
  await compareButton.click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await expect(page.locator('.comparison-choice')).toHaveCount(2)
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toBeHidden()
  await expect(compareButton).toBeFocused()

  await page.getByRole('button', { name: 'Cambiar a modo oscuro' }).click()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
})

test('keeps the compact header on one row', async ({ page }) => {
  await page.setViewportSize({ width: 640, height: 844 })

  await expect(page.locator('.privacy-note')).toBeHidden()
  await expect(page.locator('.topbar')).toHaveCSS('height', '64px')
  await expect(page.getByRole('button', { name: 'How AI picked' })).toHaveCSS('width', '34px')
})

test('keeps earlier preferences during conversational refinement', async ({ page }) => {
  await askAgent(page, 'I do not want horror')
  await askAgent(page, 'Something darker, under 90 minutes')

  await expect(page.locator('.conversation-message')).toHaveCount(4)
  await expect(page.locator('.agent-interpretation')).toContainText('No Horror')
  await expect(page.locator('.agent-interpretation')).toContainText('Twisty')
  await expect(page.locator('.agent-interpretation')).toContainText('Under 90 min')
})

test('runs a working Maya and Leo example without roadmap numbering', async ({ page }) => {
  await expect(page.locator('.onboarding')).toContainText('Maya')
  await expect(page.locator('.onboarding')).toContainText('Leo')
  await expect(page.locator('.onboarding')).toContainText('English sci-fi · under 100 min')

  await page.getByRole('button', { name: 'Run this demo' }).click({ force: true })

  await expect(page.locator('.conversation-message')).toHaveCount(2)
  await expect(page.locator('.conversation-message.user')).toContainText('English sci-fi')
  await expect(page.locator('.people-strip')).toContainText('Maya')
  await expect(page.locator('.people-strip')).toContainText('Leo')
  await expect(page.locator('.agent-interpretation')).toContainText('Sci-Fi')
  await expect(page.locator('.agent-interpretation')).toContainText('Under 100 min')
  await expect(page.locator('.agent-interpretation')).not.toContainText('0 matches now')
  await expect(page.locator('.hero-pick')).toBeVisible()
  await expect(page.locator('.graph-panel')).toBeVisible()
  await expect(page.locator('.graph-panel .eyebrow')).not.toContainText('03')
  await expect(page.locator('.results-panel .eyebrow')).not.toContainText('04')
  expect(await page.locator('.results-panel').evaluate((results) => results.getBoundingClientRect().left)).toBeLessThan(
    await page.locator('.graph-panel').evaluate((graph) => graph.getBoundingClientRect().left),
  )

  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(await page.evaluate(() => document.documentElement.clientWidth))
  expect(await page.locator('.results-panel').evaluate((results) => results.getBoundingClientRect().top)).toBeLessThan(
    await page.locator('.graph-panel').evaluate((graph) => graph.getBoundingClientRect().top),
  )
})

test('shows server semantic evidence on the selected recommendation', async ({ page }) => {
  await answerAgentTurns(page, () => ({
    preferences: { genres: ['Mystery'], excludedGenres: [], platforms: [], moods: ['Twisty'], lastUpdatedAt: Date.now() },
    recommendations: [{
      movie: { id: 'coherence-2013', title: 'Coherence', year: 2013, runtime: 89, genres: ['Mystery', 'Sci-Fi'], platforms: ['Filmin'], rating: 3.8 },
      score: 88,
      matchedPreferences: ['Mystery'],
      explanation: 'Coherence fits the mystery request. Semantic signals: witty, mysterious, playful.',
      semanticScore: 82,
      semanticTerms: ['witty', 'mysterious', 'playful'],
    }],
    meta: { extractionMode: 'model', modelStatus: 'succeeded', durationMs: 120, trace: completedTrace(fullTurn) },
  }))

  await askAgent(page, 'Witty and mysterious')

  await expect(page.locator('.hero-pick .reason')).toContainText('Semantic signals: witty, mysterious, playful')
})

test('asks before loosening a hard constraint and sends the answer to the agent', async ({ page }) => {
  const preferences = { genres: ['Mystery'], excludedGenres: [], platforms: [], moods: [], languages: ['ko'], yearRange: [1990, 1999], lastUpdatedAt: Date.now() }
  const decisions: unknown[] = []
  await answerAgentTurns(page, ({ decision }) => {
    decisions.push(decision)
    return decision === 'accept'
      ? {
          preferences: { ...preferences, yearRange: undefined },
          recommendations: [{
            movie: { id: 'decision-2022', title: 'Decision to Leave', year: 2022, runtime: 138, genres: ['Romance', 'Mystery'], platforms: ['MUBI', 'Filmin'], rating: 4.0 },
            score: 80, matchedPreferences: ['Mystery'], explanation: 'Decision to Leave is a Korean mystery.', semanticScore: 0, semanticTerms: [],
          }],
          meta: { ...deterministicMeta, trace: completedTrace(['propose_relaxation', ...fullTurn.slice(1)]) },
        }
      : {
          preferences,
          recommendations: [],
          pause: { type: 'relaxation', constraint: 'yearRange', from: [1990, 1999], to: null },
          meta: { ...deterministicMeta, trace: completedTrace(fullTurn.slice(0, 2)) },
        }
  })

  await askAgent(page, 'A Korean mystery from the 1990s')

  const proposal = page.locator('.relaxation-proposal')
  await expect(proposal).toContainText('Allow any year instead of 1990–1999?')
  await proposal.getByRole('button', { name: 'Accept' }).click()

  await expect(proposal).toBeHidden()
  await expect(page.locator('.hero-pick')).toContainText('Decision to Leave')
  expect(decisions).toEqual([undefined, 'accept'])
})

test('says nothing fits once every relaxation is rejected', async ({ page }) => {
  const preferences = { genres: [], excludedGenres: ['Horror'], platforms: [], moods: [], maxRuntime: 90, lastUpdatedAt: Date.now() }
  const meta = { ...deterministicMeta, trace: [] }
  await answerAgentTurns(page, ({ decision }) => decision === 'reject'
    ? { preferences, recommendations: [], nothingFits: { tightestConstraint: 'maxRuntime' }, meta }
    : { preferences, recommendations: [], pause: { type: 'relaxation', constraint: 'maxRuntime', from: 90, to: 105 }, meta })

  await askAgent(page, 'No horror, under 90 minutes')
  await page.locator('.relaxation-proposal').getByRole('button', { name: 'Keep it' }).click()

  await expect(page.locator('.relaxation-proposal')).toBeHidden()
  await expect(page.locator('.conversation-message.agent').last()).toContainText('The tightest one is the maximum runtime.')
})

test('publishes Reel Circle social preview metadata', async ({ page }) => {
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute('href', '/favicon.svg')
  await expect(page.locator('meta[property="og:title"]')).toHaveAttribute('content', 'Reel Circle — Find tonight\'s film')
  await expect(page.locator('meta[property="og:image"]')).toHaveAttribute('content', 'https://reel-circle-one.vercel.app/social-preview.png')
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute('content', 'summary_large_image')
})