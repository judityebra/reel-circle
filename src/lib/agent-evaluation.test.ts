import { describe, expect, it } from 'vitest'
import { evaluateAgentTrace, scoreAgentRun } from './agent-evaluation'
import type { AgentTraceEntry } from './decision-agent'

const trace: AgentTraceEntry[] = [
  { id: '1', runId: 'run-a', step: 1, tool: 'parse_constraints', rationale: 'Parse', observation: 'Done', status: 'completed', timestamp: 100 },
  { id: '2', runId: 'run-a', step: 2, tool: 'train_gcn', rationale: 'Train', observation: 'Done', status: 'completed', timestamp: 180 },
  { id: '3', runId: 'run-a', step: 3, tool: 'propose_pick', rationale: 'Propose', observation: 'Human approved.', status: 'completed', requiresApproval: true, timestamp: 260 },
  { id: '4', runId: 'run-b', step: 1, tool: 'parse_constraints', rationale: 'Parse', observation: 'Done', status: 'completed', timestamp: 400 },
  { id: '5', runId: 'run-b', step: 2, tool: 'ask_preference', rationale: 'Ask', observation: 'Waiting', status: 'waiting', requiresUserInput: true, timestamp: 450 },
]

describe('agent evaluation', () => {
  it('calculates run, efficiency, intervention, and approval metrics', () => {
    expect(evaluateAgentTrace(trace)).toEqual({
      runs: 2,
      steps: 5,
      averageSteps: 2.5,
      completedStepRate: 80,
      interventionRate: 100,
      approvalRate: 100,
      averageLatencyMs: 105,
      toolUsage: { parse_constraints: 2, train_gcn: 1, propose_pick: 1, ask_preference: 1 },
    })
  })

  it('scores safe, efficient, completed runs higher than waiting runs', () => {
    expect(scoreAgentRun(trace.filter(({ runId }) => runId === 'run-a'))).toBeGreaterThan(
      scoreAgentRun(trace.filter(({ runId }) => runId === 'run-b')),
    )
  })

  it('groups legacy traces by step-one boundaries', () => {
    const legacy = trace.map((entry) => ({ ...entry, runId: undefined }))

    expect(evaluateAgentTrace(legacy).runs).toBe(2)
  })
})