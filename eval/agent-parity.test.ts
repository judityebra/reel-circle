import { readFileSync } from 'node:fs'
import { beforeAll, describe, expect, it } from 'vitest'
import { replayConversation, resetToDeterministicEnvironment, type ParityFixtures } from './agent-parity'

const fixtures = JSON.parse(readFileSync(new URL('../fixtures/agent-parity.json', import.meta.url), 'utf8')) as ParityFixtures

beforeAll(resetToDeterministicEnvironment)

describe('recommendation agent parity fixtures', () => {
  it('cover English and Spanish, multi-turn conversations, every hard constraint and vibe phrases', () => {
    const conversations = fixtures.conversations
    expect(conversations.some(({ lang }) => lang === 'en')).toBe(true)
    expect(conversations.some(({ lang }) => lang === 'es')).toBe(true)
    expect(conversations.some(({ turns }) => turns.length > 1)).toBe(true)
    expect(conversations.filter(({ vibe }) => vibe).length).toBeGreaterThanOrEqual(3)
    const finalPreferences = conversations.flatMap(({ turns }) => turns.map(({ expected }) => expected.preferences))
    expect(finalPreferences.some(({ maxRuntime }) => maxRuntime)).toBe(true)
    expect(finalPreferences.some(({ excludedGenres }) => excludedGenres.length)).toBe(true)
    expect(finalPreferences.some(({ yearRange }) => yearRange)).toBe(true)
    expect(finalPreferences.some(({ languages }) => languages?.length)).toBe(true)
    expect(finalPreferences.some(({ genres }) => genres?.length)).toBe(true)
    expect(finalPreferences.some(({ platforms }) => platforms?.length)).toBe(true)
  })

  it('include conversations where some movies survive the hard constraints and some where none do', () => {
    const lastTurns = fixtures.conversations.map(({ turns }) => turns.at(-1)!)
    expect(lastTurns.some(({ expected }) => expected.recommendations.length > 0)).toBe(true)
    expect(lastTurns.some(({ expected }) => expected.recommendations.length === 0)).toBe(true)
  })

  it('give semantic retrieval something to match in most vibe conversations', () => {
    const vibeWithSignal = fixtures.conversations.filter(({ vibe, turns }) =>
      vibe && turns.at(-1)!.expected.recommendations.some(({ semanticScore }) => semanticScore > 0))
    expect(vibeWithSignal.length).toBeGreaterThanOrEqual(5)
  })

  it.each(fixtures.conversations.map((conversation) => [conversation.id, conversation] as const))(
    '%s is reproduced by the current TypeScript pipeline',
    async (_id, conversation) => {
      const replayed = await replayConversation(conversation, fixtures)
      expect(replayed).toEqual(conversation.turns.map(({ expected }) => expected))
    },
  )
})
