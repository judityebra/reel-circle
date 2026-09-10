import { describe, expect, it, vi } from 'vitest'
import { decideNextAction, runDecisionAgent, type AgentContext, type AgentTools } from './decision-agent'

const readyContext: AgentContext = {
  goal: 'Pick a fair movie under two hours',
  candidateCount: 5,
  labeledExamples: 6,
  modelStatus: 'ready',
  disagreement: 12,
  topMovieId: 'coherence',
  topMovieTitle: 'Coherence',
}

describe('decision agent policy', () => {
  it('parses a new goal before taking recommendation actions', () => {
    expect(decideNextAction({ ...readyContext, constraintsApplied: false })).toMatchObject({ tool: 'parse_constraints' })
  })

  it('asks permission before relaxing constraints with no candidates', () => {
    expect(decideNextAction({ ...readyContext, constraintsApplied: true, candidateCount: 0 })).toMatchObject({ tool: 'request_relaxation', requiresApproval: true })
  })

  it('asks an informative preference question when training data is sparse', () => {
    expect(decideNextAction({ ...readyContext, constraintsApplied: true, labeledExamples: 2, modelStatus: 'untrained' })).toMatchObject({ tool: 'ask_preference', requiresUserInput: true })
  })

  it('trains and compares models before proposing a final pick', () => {
    expect(decideNextAction({ ...readyContext, constraintsApplied: true, modelStatus: 'untrained' })).toMatchObject({ tool: 'train_gcn' })
    expect(decideNextAction({ ...readyContext, constraintsApplied: true, modelStatus: 'ready', modelsCompared: false })).toMatchObject({ tool: 'compare_models' })
    expect(decideNextAction({ ...readyContext, constraintsApplied: true, modelsCompared: true })).toMatchObject({ tool: 'propose_pick', requiresApproval: true })
  })

  it('executes tools until reaching a human approval gate', async () => {
    const tools: AgentTools = {
      parseConstraints: vi.fn(async () => undefined),
      trainGcn: vi.fn(async () => undefined),
      compareModels: vi.fn(async () => undefined),
    }

    const result = await runDecisionAgent({ ...readyContext, constraintsApplied: false, modelStatus: 'untrained', modelsCompared: false }, tools)

    expect(result.status).toBe('awaiting-approval')
    expect(result.trace.map(({ tool }) => tool)).toEqual(['parse_constraints', 'train_gcn', 'compare_models', 'propose_pick'])
    expect(new Set(result.trace.map(({ runId }) => runId)).size).toBe(1)
    expect(tools.trainGcn).toHaveBeenCalledOnce()
  })
})