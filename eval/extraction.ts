/**
 * Preference-extraction evaluation over hand-written EN/ES prompts.
 *
 *   npm run eval:extraction                         # deterministic parser (+ model if OPENAI_API_KEY is set)
 *   npm run eval:extraction -- --gate 0.94          # exit 1 if deterministic field accuracy < 0.94
 *   npm run eval:extraction -- --gate-exact 0.65    # exit 1 if deterministic case exact match < 0.65
 *   npm run eval:extraction -- --no-model           # skip model mode even when a key is configured
 *   npm run eval:extraction -- --verbose            # print every failing case
 */
import { createConversationPreferences, updateConversationPreferences, type ConversationPreferences } from '../src/lib/constraints'
import { extractionCases, type ExtractionCase } from './extraction-cases'
import { FIELDS, scoreCase, summarize, type CaseScore, type ExtractionSummary } from './extraction-scoring'

type Extractor = (current: ConversationPreferences, turn: string, now: number) => Promise<ConversationPreferences>

const deterministic: Extractor = async (current, turn, now) => updateConversationPreferences(current, turn, now)

/** Mirrors api/agent.ts: model extraction merged into the state, deterministic fallback per turn. */
async function modelExtractor(): Promise<{ extract: Extractor; failures: () => number }> {
  const { extractWithModel, mergeModelExtraction } = await import('../api/agent')
  let failures = 0
  return {
    failures: () => failures,
    extract: async (current, turn, now) => {
      try {
        const extraction = await extractWithModel(turn, current)
        if (extraction) return mergeModelExtraction(current, extraction, now)
      } catch {
        // Fall through to the deterministic parser exactly like the serverless handler.
      }
      failures += 1
      return updateConversationPreferences(current, turn, now)
    },
  }
}

async function runCases(extract: Extractor) {
  const results: Array<{ testCase: ExtractionCase; predicted: ConversationPreferences; score: CaseScore }> = []
  for (const testCase of extractionCases) {
    let state = createConversationPreferences(0)
    for (const [index, turn] of testCase.turns.entries()) state = await extract(state, turn, index + 1)
    results.push({ testCase, predicted: state, score: scoreCase(testCase.expected, state) })
  }
  return results
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

function report(label: string, results: Awaited<ReturnType<typeof runCases>>, verbose: boolean): ExtractionSummary {
  const summary = summarize(results.map(({ score }) => score))
  const byLang = (lang: 'en' | 'es') => summarize(results.filter(({ testCase }) => testCase.lang === lang).map(({ score }) => score))
  const multi = summarize(results.filter(({ testCase }) => testCase.turns.length > 1).map(({ score }) => score))
  const lines = [
    `## ${label}`,
    '',
    `${summary.cases} cases (${results.filter(({ testCase }) => testCase.lang === 'en').length} EN, ${results.filter(({ testCase }) => testCase.lang === 'es').length} ES, ${results.filter(({ testCase }) => testCase.turns.length > 1).length} multi-turn)`,
    '',
    '| Metric | All | EN | ES | Multi-turn |',
    '| --- | --- | --- | --- | --- |',
    `| Field accuracy (micro) | ${percent(summary.overallFieldAccuracy)} | ${percent(byLang('en').overallFieldAccuracy)} | ${percent(byLang('es').overallFieldAccuracy)} | ${percent(multi.overallFieldAccuracy)} |`,
    `| Case exact match | ${percent(summary.exactMatch)} | ${percent(byLang('en').exactMatch)} | ${percent(byLang('es').exactMatch)} | ${percent(multi.exactMatch)} |`,
    `| Exclusion recall | ${percent(summary.exclusionRecall)} | ${percent(byLang('en').exclusionRecall)} | ${percent(byLang('es').exclusionRecall)} | ${percent(multi.exclusionRecall)} |`,
    `| Contradictions | ${percent(summary.contradictionRate)} | ${percent(byLang('en').contradictionRate)} | ${percent(byLang('es').contradictionRate)} | ${percent(multi.contradictionRate)} |`,
    '',
    '| Field | Accuracy |',
    '| --- | --- |',
    ...FIELDS.map((field) => `| ${field} | ${percent(summary.fieldAccuracy[field])} |`),
    '',
  ]
  if (verbose) {
    lines.push('Failing cases:', '')
    results.filter(({ score }) => !score.exactMatch).forEach(({ testCase, predicted, score }) => {
      const wrong = FIELDS.filter((field) => !score.fields[field])
      const shown = Object.fromEntries(wrong.map((field) => [field, { expected: testCase.expected[field] ?? null, got: predicted[field] ?? null }]))
      lines.push(`- ${testCase.id}: ${JSON.stringify(shown)}`)
    })
    lines.push('')
  }
  console.log(lines.join('\n'))
  return summary
}

async function main() {
  const args = process.argv.slice(2)
  const flagValue = (flag: string) => {
    const index = args.indexOf(flag)
    if (index < 0) return undefined
    const value = Number(args[index + 1])
    if (Number.isNaN(value) || value < 0 || value > 1) throw new Error(`${flag} needs a number between 0 and 1`)
    return value
  }
  const gates: Array<[string, number | undefined, (summary: ExtractionSummary) => number]> = [
    ['field accuracy', flagValue('--gate'), (summary) => summary.overallFieldAccuracy],
    ['case exact match', flagValue('--gate-exact'), (summary) => summary.exactMatch],
  ]
  const verbose = args.includes('--verbose')
  const useModel = Boolean(process.env.OPENAI_API_KEY) && !args.includes('--no-model')

  console.log('# Preference extraction evaluation\n')
  const baseline = report('Deterministic parser (updateConversationPreferences)', await runCases(deterministic), verbose)

  if (useModel) {
    const model = await modelExtractor()
    report(`Model extraction (${process.env.OPENAI_MODEL ?? 'gpt-4.1-mini'}) via extractWithModel + mergeModelExtraction`, await runCases(model.extract), verbose)
    console.log(`Model turns that fell back to the deterministic parser: ${model.failures()}\n`)
  } else {
    console.log('Model mode skipped (set OPENAI_API_KEY to enable).\n')
  }

  gates.forEach(([name, threshold, metric]) => {
    if (threshold === undefined) return
    const value = metric(baseline)
    const passed = value >= threshold
    console.log(`Gate: deterministic ${name} ${percent(value)} ${passed ? '>=' : '<'} ${percent(threshold)} -> ${passed ? 'PASS' : 'FAIL'}`)
    if (!passed) process.exitCode = 1
  })
}

await main()
