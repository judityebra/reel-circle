import { expect, test } from '@playwright/test'

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
  await expect(page.getByRole('button', { name: 'Open AI lab' })).toHaveCSS('width', '34px')
})

test('keeps earlier preferences during conversational refinement', async ({ page }) => {
  await page.locator('#constraint-prompt').fill('I do not want horror')
  await page.getByRole('button', { name: 'Apply', exact: true }).click()
  await page.locator('#constraint-prompt').fill('Something darker, under 90 minutes')
  await page.getByRole('button', { name: 'Apply', exact: true }).click()

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
  await page.route('**/api/agent', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        preferences: { genres: ['Mystery'], excludedGenres: [], platforms: [], moods: ['Twisty'], lastUpdatedAt: Date.now() },
        recommendations: [{
          movie: { id: 'coherence-2013', title: 'Coherence', year: 2013, runtime: 89, genres: ['Mystery', 'Sci-Fi'], platforms: ['Filmin'], rating: 3.8 },
          score: 88,
          matchedPreferences: ['Mystery'],
          explanation: 'Coherence fits the mystery request. Semantic signals: witty, mysterious, playful.',
          semanticScore: 82,
          semanticTerms: ['witty', 'mysterious', 'playful'],
        }],
        meta: {
          extractionMode: 'model', modelStatus: 'succeeded', durationMs: 120,
          trace: ['extract_preferences', 'filter_candidates', 'semantic_retrieval', 'rank_candidates', 'validate_results'].map((tool) => ({ tool, status: 'completed', durationMs: 1, observation: 'ok' })),
        },
      }),
    })
  })

  await page.locator('#constraint-prompt').fill('Witty and mysterious')
  await page.getByRole('button', { name: 'Apply', exact: true }).click()

  await expect(page.locator('.hero-pick .reason')).toContainText('Semantic signals: witty, mysterious, playful')
})

test('publishes Reel Circle social preview metadata', async ({ page }) => {
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute('href', '/favicon.svg')
  await expect(page.locator('meta[property="og:title"]')).toHaveAttribute('content', 'Reel Circle — Find tonight\'s film')
  await expect(page.locator('meta[property="og:image"]')).toHaveAttribute('content', 'https://reel-circle-one.vercel.app/social-preview.png')
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute('content', 'summary_large_image')
})