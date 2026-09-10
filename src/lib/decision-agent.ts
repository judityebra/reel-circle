export type AgentTool =
  | 'parse_constraints'
  | 'inspect_shortlist'
  | 'ask_preference'
  | 'train_gcn'
  | 'compare_models'
  | 'request_relaxation'
  | 'propose_pick'

export interface AgentContext {
  goal: string
  constraintsApplied?: boolean
  candidateCount: number
  labeledExamples: number
  modelStatus: 'untrained' | 'ready' | 'stale'
  modelsCompared?: boolean
  disagreement: number
  uncertaintyResolved?: boolean
  topMovieId?: string
  topMovieTitle?: string
}

export interface AgentAction {
  tool: AgentTool
  rationale: string
  requiresApproval?: boolean
  requiresUserInput?: boolean
}

export interface AgentTraceEntry extends AgentAction {
  id: string
  runId?: string
  step: number
  status: 'completed' | 'waiting'
  observation: string
  timestamp: number
}

export interface AgentTools {
  parseConstraints?: (goal: string) => Promise<void>
  trainGcn?: () => Promise<void>
  compareModels?: () => Promise<void>
}

export interface AgentRunResult {
  status: 'awaiting-input' | 'awaiting-approval' | 'completed' | 'max-steps'
  trace: AgentTraceEntry[]
  context: AgentContext
}

export function decideNextAction(context: AgentContext): AgentAction {
  if (!context.constraintsApplied && context.goal.trim()) {
    return { tool: 'parse_constraints', rationale: 'Translate the stated goal into executable filters.' }
  }
  if (context.candidateCount === 0) {
    return { tool: 'request_relaxation', rationale: 'No candidate satisfies every hard constraint.', requiresApproval: true }
  }
  if (context.labeledExamples < 4) {
    return { tool: 'ask_preference', rationale: `The learner has ${context.labeledExamples}/4 required labels.`, requiresUserInput: true }
  }
  if (context.disagreement > 30 && !context.uncertaintyResolved) {
    return { tool: 'ask_preference', rationale: `Group disagreement is ${context.disagreement} points; one comparison can reduce uncertainty.`, requiresUserInput: true }
  }
  if (context.modelStatus !== 'ready') {
    return { tool: 'train_gcn', rationale: 'Enough labels exist to refresh the graph neural model.' }
  }
  if (!context.modelsCompared) {
    return { tool: 'compare_models', rationale: 'Measure the GCN against the deterministic baseline before using it.' }
  }
  return { tool: 'propose_pick', rationale: `${context.topMovieTitle ?? 'The top candidate'} leads after constraints, fairness, and model comparison.`, requiresApproval: true }
}

export async function runDecisionAgent(
  initialContext: AgentContext,
  tools: AgentTools,
  maxSteps = 8,
): Promise<AgentRunResult> {
  let context = { ...initialContext }
  const trace: AgentTraceEntry[] = []
  const runId = crypto.randomUUID()

  for (let index = 0; index < maxSteps; index += 1) {
    const action = decideNextAction(context)
    const waiting = Boolean(action.requiresApproval || action.requiresUserInput)
    let observation = action.rationale

    if (!waiting && action.tool === 'parse_constraints') {
      await tools.parseConstraints?.(context.goal)
      context = { ...context, constraintsApplied: true }
      observation = 'Goal converted into structured constraints.'
    } else if (!waiting && action.tool === 'train_gcn') {
      await tools.trainGcn?.()
      context = { ...context, modelStatus: 'ready' }
      observation = 'GCN training completed and predictions are available.'
    } else if (!waiting && action.tool === 'compare_models') {
      await tools.compareModels?.()
      context = { ...context, modelsCompared: true }
      observation = 'Baseline and GCN rankings were compared.'
    }

    trace.push({
      ...action,
      id: `${Date.now()}-${index}-${action.tool}`,
      runId,
      step: index + 1,
      status: waiting ? 'waiting' : 'completed',
      observation,
      timestamp: Date.now(),
    })

    if (action.requiresUserInput) return { status: 'awaiting-input', trace, context }
    if (action.requiresApproval) return { status: 'awaiting-approval', trace, context }
  }

  return { status: 'max-steps', trace, context }
}