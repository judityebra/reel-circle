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

test('publishes Reel Circle social preview metadata', async ({ page }) => {
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute('href', '/favicon.svg')
  await expect(page.locator('meta[property="og:title"]')).toHaveAttribute('content', 'Reel Circle — Find tonight\'s film')
  await expect(page.locator('meta[property="og:image"]')).toHaveAttribute('content', 'https://reel-circle-one.vercel.app/social-preview.png')
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute('content', 'summary_large_image')
})