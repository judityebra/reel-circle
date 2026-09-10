import type { AgentTool, AgentTraceEntry } from './decision-agent'

export interface AgentEvaluation {
  runs: number
  steps: number
  averageSteps: number
  completedStepRate: number
  interventionRate: number
  approvalRate: number
  averageLatencyMs: number
  toolUsage: Partial<Record<AgentTool, number>>
}

const groupByRun = (trace: AgentTraceEntry[]) => {
  const groups = new Map<string, AgentTraceEntry[]>()
  let legacyRun = -1
  trace.forEach((entry) => {
    if (!entry.runId && (entry.step === 1 || legacyRun < 0)) legacyRun += 1
    const runId = entry.runId ?? `legacy-${legacyRun}`
    groups.set(runId, [...(groups.get(runId) ?? []), entry])
  })
  return [...groups.values()]
}

export function evaluateAgentTrace(trace: AgentTraceEntry[]): AgentEvaluation {
  const runs = groupByRun(trace)
  const completed = trace.filter(({ status }) => status === 'completed').length
  const interventionRuns = runs.filter((run) => run.some(({ requiresApproval, requiresUserInput }) => requiresApproval || requiresUserInput)).length
  const resolvedApprovals = trace.filter(({ requiresApproval, observation }) => requiresApproval && /Human (approved|rejected)/.test(observation))
  const approved = resolvedApprovals.filter(({ observation }) => observation.includes('Human approved')).length
  const latencies = runs.map((run) => Math.max(...run.map(({ timestamp }) => timestamp)) - Math.min(...run.map(({ timestamp }) => timestamp)))
  const toolUsage = trace.reduce<Partial<Record<AgentTool, number>>>((usage, { tool }) => ({ ...usage, [tool]: (usage[tool] ?? 0) + 1 }), {})

  return {
    runs: runs.length,
    steps: trace.length,
    averageSteps: runs.length ? Number((trace.length / runs.length).toFixed(1)) : 0,
    completedStepRate: trace.length ? Math.round((completed / trace.length) * 100) : 0,
    interventionRate: runs.length ? Math.round((interventionRuns / runs.length) * 100) : 0,
    approvalRate: resolvedApprovals.length ? Math.round((approved / resolvedApprovals.length) * 100) : 0,
    averageLatencyMs: latencies.length ? Math.round(latencies.reduce((total, value) => total + value, 0) / latencies.length) : 0,
    toolUsage,
  }
}

export function scoreAgentRun(trace: AgentTraceEntry[]): number {
  if (trace.length === 0) return 0
  const completedRate = trace.filter(({ status }) => status === 'completed').length / trace.length
  const safeGate = trace.some(({ requiresApproval, requiresUserInput }) => requiresApproval || requiresUserInput) ? 1 : 0.75
  const boundedEfficiency = Math.max(0, 1 - Math.max(0, trace.length - 4) * 0.08)
  return Math.round((completedRate * 0.55 + safeGate * 0.25 + boundedEfficiency * 0.2) * 100)
}