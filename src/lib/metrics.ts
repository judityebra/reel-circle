import type { FeedbackEvent } from './session'

export interface LearningMetrics {
  interactions: number
  accepted: number
  skipped: number
  comparisons: number
  acceptanceRate: number
}

export function summarizeFeedback(events: FeedbackEvent[]): LearningMetrics {
  const accepted = events.filter(({ action }) => action === 'pick' || action === 'love').length
  const skipped = events.filter(({ action }) => action === 'skip').length
  const comparisons = events.filter(({ action }) => action === 'comparison').length
  return {
    interactions: events.length,
    accepted,
    skipped,
    comparisons,
    acceptanceRate: events.length === 0 ? 0 : Math.round((accepted / events.length) * 100),
  }
}