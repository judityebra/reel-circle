# Evaluation

Reel Circle has two offline evaluations in [`eval/`](../eval):

| Command | What it measures |
| --- | --- |
| `npm run eval:recs` | Ranking quality of every recommender in the app on a seeded synthetic world, and group-fairness outcomes of the four aggregation strategies. |
| `npm run eval:extraction` | How well conversation turns (EN/ES, single and multi-turn) are turned into structured preferences. |

Both are deterministic. `eval:recs` replaces `Math.random` with a seeded generator, which also makes TensorFlow.js batch shuffling reproducible: two runs with the same flags produce byte-identical output. The metric functions have unit tests (`eval/*.test.ts`, run by `npm test`).

All numbers below come from the default configuration (`seed=2026`). The recommendation benchmark takes about 50 seconds on a laptop CPU.

## 1. Recommendation benchmark

### Setup

**Synthetic world** ([`eval/synthetic.ts`](../eval/synthetic.ts)):

- **Directors (30):** each has a home genre, a home language and a talent offset.
- **Movies (150):** each takes its director's home genre 85% of the time, plus 0–2 extra genres. Each also gets moods correlated with its genres, its director's language 80% of the time, a year (1960–2024), a runtime, and a latent quality. The community rating is `3.3 + 0.45·quality + noise`.
- **Users (60):** each has latent affinities for genres (two favourites, two dislikes), moods, languages and directors. They also have a preferred era, a runtime comfort level, a personal weight on quality, and a bias.
- **Hidden ratings:** every user has one for every movie, in half stars. The rating is a noisy function of the latent utility. On the held-out set the mean is 3.23 and 35% are "liked" (≥ 4).

**Protocol:**

- **Held-out set:** 50 movies, fixed for all users. They are never used as labels.
- **Labels:** each user has a seeded ordering of the other 100 movies. The first 8, 16 or 32 become that user's known ratings, so smaller label sets are prefixes of larger ones.
- **Ranking:** each method scores the 50 held-out movies for one user. Ties keep the (random) held-out order.

**Metrics:**

| Metric | Definition |
| --- | --- |
| NDCG@10 | Linear gain = hidden rating. |
| Precision@10 | Share of the top 10 with a hidden rating ≥ 4. |
| Spearman | Rank correlation between predicted scores and hidden ratings over all 50 movies (tie-averaged ranks). |

Each value is the mean across users with a 95% percentile-bootstrap CI (1000 resamples).

**Methods:**

| Method | What it is |
| --- | --- |
| `random` | Uniform random scores. |
| `popularity` | Community rating. |
| `scorer` | `scoreCandidates` with `preferredGenres` derived exactly as `App.tsx` does on Letterboxd import: genres of rated catalog films, in catalog order, first three, ignoring the rating value. |
| `scorer+feedback` | The same profile, with each label replayed through `applyTasteFeedback`: ≥ 4.5 → love, ≥ 3 → pick, otherwise skip. |
| `mlp` / `gcn` | Raw predictions of `trainNeuralTaste` / `trainGraphNeuralTaste`, trained on `buildNeuralExamples` (rating / 5). |
| `mlp-hybrid` / `gcn-hybrid` | `scorer`, blended with the network at `neuralBlend = confidence / 200`, as the app does. |

### Results

**8 labels per user.** Mean model confidence: MLP 66.7 (blend 33%), GCN 56.5 (blend 28%).

| Method | NDCG@10 | Precision@10 | Spearman |
| --- | --- | --- | --- |
| random | 0.661 [0.637, 0.682] | 0.315 [0.267, 0.362] | -0.012 [-0.046, 0.022] |
| popularity | **0.838** [0.820, 0.856] | **0.585** [0.533, 0.637] | **0.393** [0.360, 0.430] |
| scorer | 0.758 [0.736, 0.781] | 0.448 [0.395, 0.512] | 0.186 [0.138, 0.236] |
| scorer+feedback | 0.797 [0.775, 0.818] | 0.505 [0.447, 0.567] | 0.330 [0.287, 0.374] |
| mlp | 0.719 [0.692, 0.743] | 0.413 [0.362, 0.465] | 0.172 [0.132, 0.213] |
| gcn | 0.741 [0.715, 0.768] | 0.448 [0.397, 0.503] | 0.189 [0.142, 0.238] |
| mlp-hybrid | 0.774 [0.750, 0.796] | 0.467 [0.407, 0.530] | 0.249 [0.207, 0.292] |
| gcn-hybrid | 0.776 [0.757, 0.797] | 0.465 [0.405, 0.522] | 0.249 [0.206, 0.290] |

**16 labels per user.** Mean model confidence: MLP 99.8 (blend 50%), GCN 98.2 (blend 49%).

| Method | NDCG@10 | Precision@10 | Spearman |
| --- | --- | --- | --- |
| random | 0.676 [0.655, 0.696] | 0.352 [0.308, 0.402] | -0.002 [-0.043, 0.041] |
| popularity | **0.838** [0.820, 0.856] | **0.585** [0.532, 0.637] | **0.393** [0.358, 0.427] |
| scorer | 0.755 [0.736, 0.774] | 0.457 [0.402, 0.510] | 0.173 [0.134, 0.219] |
| scorer+feedback | 0.817 [0.798, 0.834] | 0.568 [0.520, 0.615] | 0.379 [0.340, 0.418] |
| mlp | 0.763 [0.738, 0.788] | 0.482 [0.428, 0.535] | 0.252 [0.204, 0.293] |
| gcn | 0.781 [0.757, 0.805] | 0.507 [0.455, 0.553] | 0.258 [0.216, 0.307] |
| mlp-hybrid | 0.783 [0.763, 0.804] | 0.513 [0.463, 0.568] | 0.303 [0.267, 0.340] |
| gcn-hybrid | 0.789 [0.770, 0.810] | 0.510 [0.452, 0.563] | 0.307 [0.268, 0.347] |

**32 labels per user.** Mean model confidence: MLP 99.8 (blend 50%), GCN 97.5 (blend 49%).

| Method | NDCG@10 | Precision@10 | Spearman |
| --- | --- | --- | --- |
| random | 0.672 [0.642, 0.698] | 0.352 [0.307, 0.402] | 0.008 [-0.022, 0.037] |
| popularity | **0.838** [0.819, 0.856] | 0.585 [0.535, 0.637] | 0.393 [0.357, 0.426] |
| scorer | 0.758 [0.737, 0.776] | 0.467 [0.417, 0.520] | 0.181 [0.146, 0.213] |
| scorer+feedback | 0.822 [0.800, 0.841] | **0.595** [0.545, 0.648] | **0.416** [0.371, 0.456] |
| mlp | 0.803 [0.785, 0.822] | 0.548 [0.497, 0.600] | 0.345 [0.302, 0.383] |
| gcn | 0.807 [0.790, 0.827] | 0.563 [0.520, 0.607] | 0.340 [0.294, 0.383] |
| mlp-hybrid | 0.822 [0.804, 0.841] | 0.587 [0.535, 0.632] | 0.371 [0.337, 0.405] |
| gcn-hybrid | 0.817 [0.799, 0.834] | 0.563 [0.517, 0.612] | 0.346 [0.313, 0.382] |

### What this says

1. **Nothing personalized beats sorting by community rating on top-10 quality.**
   - Popularity leads NDCG@10 at every label count. At 32 labels, `scorer+feedback` and `mlp-hybrid` reach its NDCG and Precision only within overlapping CIs.
   - Part of this is built into the generator: latent quality drives both the community rating and every user's ratings (see *Limitations*). Even so, a personalized method should at least match popularity here, and the shipped path does not.
2. **The scorer as shipped does not learn from more ratings.**
   - `scorer` is flat (NDCG ≈ 0.76, Spearman ≈ 0.18) from 8 to 32 labels, and it ranks below popularity even though it contains the community rating.
   - The cause is how `App.tsx` derives `preferredGenres`: from *every* rated film, in catalog order, ignoring the rating. Genres of films the user disliked get the same bonus as genres they loved.
   - Deriving them from highly rated films is the obvious fix. This benchmark can measure it.
3. **Replaying ratings as skip/pick/love feedback is the best personalized method.**
   - It improves steadily with labels (Spearman 0.33 → 0.38 → 0.42).
   - It is the only method whose Spearman overtakes popularity, at 32 labels (0.416 vs 0.393, CIs overlapping).
4. **The MLP and GCN need data and are overconfident.**
   - On their own they are *worse* than the shipped scorer at 8 labels.
   - They improve with labels (NDCG ≈ 0.80 at 32). The GCN is slightly ahead of the MLP, but the CIs overlap throughout.
   - Their reported confidence measures training fit, not held-out accuracy. It saturates near 100 at 16 labels, which pins the blend at its 0.5 cap, while held-out Spearman is only about 0.25.
5. **Blending helps the shipped scorer.** Both hybrids beat `scorer` at every label count. The gain comes mostly from fixing its weak genre signal, and they do not beat `scorer+feedback`.

### Group fairness

The setup:

- **Groups:** 200 random triples of synthetic users.
- **Member profiles:** each member's `scorer+feedback` profile from 16 labels.
- **Pick:** for each strategy, the top `scoreCandidates` pick over the held-out set.
- **Scoring:** each pick is scored on the members' hidden ratings.
- **References:** a random held-out movie, and two oracles with access to the hidden ratings (best mean, best minimum).

| Strategy | Mean member rating | Min member rating | Someone rates ≤ 2.5 |
| --- | --- | --- | --- |
| random | 3.18 [3.07, 3.29] | 2.28 [2.15, 2.42] | 67.0 [60.5, 73.0]% |
| balanced | **4.10** [4.01, 4.20] | **3.38** [3.25, 3.49] | **27.5** [21.5, 34.0]% |
| average | 4.08 [3.98, 4.18] | 3.32 [3.18, 3.45] | 30.0 [23.5, 36.5]% |
| least-misery | 3.94 [3.84, 4.04] | 3.21 [3.07, 3.34] | 32.5 [26.5, 39.0]% |
| nash | 4.07 [3.97, 4.16] | 3.33 [3.19, 3.46] | 28.5 [22.5, 35.0]% |
| oracle (max mean) | 4.73 [4.70, 4.76] | 4.33 [4.25, 4.41] | 0.0 [0.0, 0.0]% |
| oracle (max min) | 4.72 [4.68, 4.75] | 4.38 [4.31, 4.44] | 0.0 [0.0, 0.0]% |

- **Better than random, far from the oracle.** Every strategy cuts the chance that someone in the group dislikes the pick from 67% to roughly 28–33%. That is still far from the oracle's 0%: the bottleneck is individual preference estimates, not aggregation.
- **The strategies are statistically indistinguishable.** Their CIs overlap on every column.
- **Least-misery is, if anything, worst on the minimum rating**, the thing it is meant to protect.
  - Ties do not explain it: on average only ~1.1 movies tie for the top score under any strategy.
  - The likely cause is that taking the minimum of *noisy predicted* scores mostly selects for the member whose estimate is noisiest.
  - Averaging (`balanced`, `average`, `nash`) is more robust to that noise.
  - With weak individual models, least-misery should not be presented as the "fair" option without this caveat.

## 2. Preference extraction

### Cases and metrics

[`eval/extraction-cases.ts`](../eval/extraction-cases.ts) has 52 hand-written prompts:

- 31 English and 21 Spanish.
- 10 multi-turn: refinements, persistence, "start over", and lifting an exclusion.

**Annotation.** Each case is labelled with the state a user would expect after the last turn, following the annotation guide at the top of that file. The cases and labels were written before running the parser and have not been edited to match its output. Ambiguous phrasings ("with 4 friends": 4 or 5 people? "short": how short?) were left out instead of being labelled to suit the parser.

**Fields checked in every case.** Anything not labelled must be empty.

- genres
- excluded genres
- moods
- platforms
- max runtime
- year range (open-ended ranges accept any end ≥ 2024)
- languages
- people count
- viewing context

**Metrics:**

| Metric | Definition |
| --- | --- |
| Field accuracy | Share of (case, field) checks that are correct. Inflated by the many fields correctly left empty. |
| Case exact match | All nine fields correct. The stricter and more informative number. |
| Exclusion recall | Share of genres the user ruled out that end up in `excludedGenres`. |
| Contradictions | Share of cases where a genre is both requested and excluded, or a genre the user ruled out shows up as requested. |

**Modes:**

- **Deterministic:** `updateConversationPreferences`, applied turn by turn.
- **Model:** runs when `OPENAI_API_KEY` is set. It mirrors `api/agent.ts`: `extractWithModel`, then `mergeModelExtraction`, with deterministic fallback on failure. The report counts fallbacks.

### Results (deterministic parser)

| Metric | All | EN | ES | Multi-turn |
| --- | --- | --- | --- | --- |
| Field accuracy | 95.1% | 93.5% | 97.4% | 96.7% |
| Case exact match | 69.2% | 61.3% | 81.0% | 80.0% |
| Exclusion recall | 53.8% | 50.0% | 60.0% | 100.0% |
| Contradictions | 9.6% | 12.9% | 4.8% | 0.0% |

| Field | genres | excludedGenres | moods | platforms | maxRuntime | yearRange | languages | peopleCount | viewingContext |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 90.4% | 90.4% | 98.1% | 98.1% | 96.2% | 94.2% | 98.1% | 90.4% | 100.0% |

The 16 failing cases (`npm run eval:extraction -- --verbose`) fall into clear groups:

- **Negation the parser does not recognise.** This is the most serious group, because the excluded genre is *requested* instead (5 cases, all the contradictions). Examples: "nobody likes horror", "anything but comedy", "not in the mood for drama", the second genre in "without sci-fi **or horror**", and "que no sea de animación **ni de terror**".
- **Runtime caps:** "nothing over two hours" and "at most 95 minutes" are missed.
- **A false platform:** "an hour and a half **max**" becomes HBO Max.
- **Decades and lower bounds:** "the 90s", "de los 90" and "nothing older than 2000" are missed. ("1990s" is handled.)
- **Group size:** "We're 4 people", "me and two friends", "five people" and "somos tres" are missed. Only "with/for N people|friends" patterns are recognised.
- **Inflection:** "coreana" is not matched (only "coreano").
- **A mood from "darker":** the parser adds Twisty, which the annotation guide deliberately leaves unlabelled.

Multi-turn state handling is the parser's strongest area: "start over", lifting an exclusion with "actually comedy is fine", and exclusions persisting across turns all work.

### Model mode

No model results are reported here because the benchmark was run without an API key. Run `OPENAI_API_KEY=... npm run eval:extraction` to add them. One limitation is predictable from the code and not yet measured: `mergeModelExtraction` only overwrites fields the model returns non-empty, so a turn like "actually comedy is fine" cannot clear an earlier exclusion in model mode.

### CI gate

CI runs:

```bash
npm run eval:extraction -- --no-model --gate 0.94 --gate-exact 0.65
```

That fails the build if deterministic field accuracy drops below 94% (baseline 95.1%) or case exact match drops below 65% (baseline 69.2%, i.e. more than two newly failing cases). When the parser improves, raise the thresholds to just below the new baseline.

## Limitations

### Recommendation benchmark limitations

- **The world is synthetic and reflects my modelling choices.**
  - Utility is additive over genre, mood, language, director, era, runtime and quality. These are the same features the scorer and the networks see, which flatters feature-based models.
  - The quality effect is strong, which favours popularity.
  - There is no collaborative signal (users similar to other users) for any method to exploit, and none of the app's methods would use one.
  - Absolute numbers are not predictions of real-world performance. The value is in comparing methods and catching regressions.
- **One seed.** CIs cover sampling of users (and triples), not the variance from regenerating the world. Re-run with `--seed` to check that conclusions hold.
- **NDCG over 50 candidates is lenient.** Random already scores ~0.66. Look at the gaps between methods, not the absolute values.
- **Hidden ratings exist for every movie.** Real data is missing-not-at-random: people rate what they chose to watch. That bias is not modelled.
- **The skip/pick/love mapping is a choice.** The thresholds (≥ 4.5 / ≥ 3) were not tuned. Other mappings will move `scorer+feedback`.
- **The group eval is narrow.** It covers only triples, only `scorer+feedback` member profiles at 16 labels, no neural blending, and no runtime or platform constraints.

### Extraction eval limitations

- **The sample is small.** With 52 cases, one case is worth 1.9 points of exact match. No CIs are reported, and differences of a few points are noise.
- **One annotator wrote the cases** and knew the app's genre and mood vocabulary. Real users phrase things more loosely, and probably in more varied ways, especially in Spanish.
- **Some fields are unchecked or absent.** `rankingMode`, which the parser also extracts, is not scored because the model schema does not include it. Moods are only labelled when unambiguous, so mood accuracy is optimistic.

## Reproducing

```bash
npm run eval:recs                         # full benchmark, ~50 s
npm run eval:recs -- --users 20 --triples 50 --seed 7
npm run eval:recs -- --json recs.json     # also write machine-readable results
npm run eval:extraction -- --verbose      # list failing cases
npm test                                  # includes the metric unit tests
```
