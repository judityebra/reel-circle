/**
 * Regenerates fixtures/agent-parity.json from the current TypeScript pipeline.
 *
 *   npm run fixtures:parity
 *
 * Only regenerate when a behaviour change to the recommendation agent is intended; the diff of the
 * JSON is the review of that change.
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { buildParityFixtures, PARITY_FIXTURES_URL } from './agent-parity'

const fixtures = await buildParityFixtures()
mkdirSync(new URL('.', PARITY_FIXTURES_URL), { recursive: true })
writeFileSync(PARITY_FIXTURES_URL, `${JSON.stringify(fixtures, null, 2)}\n`)
const turns = fixtures.conversations.reduce((total, { turns }) => total + turns.length, 0)
console.log(`Wrote ${fixtures.conversations.length} conversations (${turns} turns, ${fixtures.candidatePool.length} candidate movies) to fixtures/agent-parity.json`)
