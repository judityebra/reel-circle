import { readFileSync } from 'node:fs'
import { beforeAll, describe, expect, it } from 'vitest'
import { HARD_CONSTRAINTS, PARITY_FIXTURES_URL, replayConversation, resetToDeterministicEnvironment, type ParityFixtures } from './agent-parity'

const fixtures = JSON.parse(readFileSync(PARITY_FIXTURES_URL, 'utf8')) as ParityFixtures
const allTurns = fixtures.conversations.flatMap(({ turns }) => turns)
const lastTurns = fixtures.conversations.map(({ turns }) => turns.at(-1)!)

beforeAll(resetToDeterministicEnvironment)

describe('recommendation agent parity fixtures', () => {
  it('cover English and Spanish, multi-turn conversations and vibe phrases', () => {
    expect(fixtures.conversations.some(({ lang }) => lang === 'en')).toBe(true)
    expect(fixtures.conversations.some(({ lang }) => lang === 'es')).toBe(true)
    expect(fixtures.conversations.some(({ turns }) => turns.length > 1)).toBe(true)
    expect(fixtures.conversations.filter(({ vibe }) => vibe).length).toBeGreaterThanOrEqual(3)
  })

  it('exercise every hard constraint', () => {
    const allTurnPreferences = allTurns.map(({ expected }) => expected.preferences)
    for (const constraint of HARD_CONSTRAINTS) {
      const used = allTurnPreferences.some((preferences) => {
        const value = preferences[constraint]
        return Array.isArray(value) ? value.length > 0 : Boolean(value)
      })
      expect(used, constraint).toBe(true)
    }
  })

  it('carry each turn’s incoming preferences so a port can replay it without the TypeScript code', () => {
    for (const { turns } of fixtures.conversations) {
      expect(turns[0].request.preferences).toBeNull()
      turns.slice(1).forEach((turn, index) => expect(turn.request.preferences).toEqual(turns[index].expected.preferences))
    }
  })

  it('send persons without ratings or watch history, with learned weights for more than one person', () => {
    expect(fixtures.persons.every(({ ratings, watched }) => ratings.length === 0 && watched.length === 0)).toBe(true)
    expect(fixtures.persons.filter(({ tasteWeights, featureWeights }) => tasteWeights && featureWeights).length).toBeGreaterThanOrEqual(2)
  })

  it('include conversations where some movies meet the hard constraints and some where none do', () => {
    expect(lastTurns.some(({ expected }) => expected.candidatesMeetingConstraints > 0)).toBe(true)
    expect(lastTurns.some(({ expected }) => expected.candidatesMeetingConstraints === 0)).toBe(true)
  })

  it('give semantic retrieval something to match in most vibe conversations', () => {
    const vibeWithSignal = fixtures.conversations.filter(({ vibe, turns }) =>
      vibe && turns.at(-1)!.expected.recommendations.some(({ semanticScore }) => semanticScore > 0))
    expect(vibeWithSignal.length).toBeGreaterThanOrEqual(5)
  })

  it('record no hard-constraint violations in any recommendation', () => {
    expect(allTurns.flatMap(({ expected }) => expected.violations)).toEqual([])
  })

  it.each(fixtures.conversations.map((conversation) => [conversation.id, conversation] as const))(
    '%s is reproduced by the current TypeScript pipeline',
    async (_id, conversation) => {
      const replayed = await replayConversation(conversation.turns.map(({ request }) => request.message), fixtures)
      expect(replayed).toEqual(conversation.turns)
    },
  )
})
