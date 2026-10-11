# What the judge's mistakes tell us

October 6, 2026. **The error review is complete.** The review checked the saved decisions for all 60 validation questions and read the full question and both answers for all **37 questions with a mismatch in at least one answer order**. That includes all 25 order disagreements, all six original-order regressions from the 20-update trial, and all five swapped-order regressions.

This is a **model-based diagnostic review with labels visible**, not independent human adjudication. Original labels, model weights, prompts, and measured scores are unchanged. No new model generations or training ran. Calibration and final-test records were not opened.

## Start with these three findings

1. **Position still influences the choice.** On 12 questions, the 200-update judge chooses whichever answer appears first; on two it chooses whichever appears second. Eleven more change between a tie and a preferred answer after swapping. This is observable behavior, not proof of a specific internal cause.
2. **Our tie wording is stricter than the dataset wording.** The publisher describes zero preference as “about the same”; our prompt says “equal.” All 20 validation tie labels have released individual reviewers favoring both sides, with no individual zero vote. The same is true for all 29 ties in the prepared training subset. A tie label therefore does not establish unanimous agreement or factual equivalence. This suggests a prompt mismatch worth testing; it does not prove that wording caused the errors.
3. **Preference mistakes are not all factual mistakes.** Some answers differ mainly in style. Others contain visible instruction failures, unsupported additions, or unverified domain claims. Matching the publisher label cannot certify an answer's truth.

The tie definition comes from the saved publisher [dataset card](<../Training Data/raw/helpsteer2/DATASET_CARD.md>) under Preferences. The released votes were inspected without recomputing or changing labels. The released closest-preference list contains fewer than three entries for six validation ties and seven training-subset ties, so reconstructing the publisher's processing from that field would be unsafe.

## What happens after what in this review?

1. Verify the original baseline and both trained-run receipts.
2. Join each question's original and swapped predictions by its saved ID.
3. Translate swapped A/B choices back to the original answer identity. A displayed A can refer to original B after a swap.
4. Count regressions and disagreements across the same questions.
5. Read every error question and both answers; keep observations separate from the publisher label.
6. Inspect released reviewer preferences. Check simple constraints directly and use primary sources for two specific factual checks.
7. Save one note for each of the 37 error questions, the full 60-question prediction table, and this report.
8. Choose one small next experiment and its limits before running it. That proposed experiment has **not run**.

## How answer position affects the result

| Behavior across both orders | Baseline | 20 updates | 200 updates |
| --- | ---: | ---: | ---: |
| Chooses whichever answer is first | 21 | 8 | 12 |
| Chooses whichever answer is second | 2 | 1 | 2 |
| Changes between tie and a winner | 7 | 19 | 11 |
| Total questions with different underlying choices | 30 | 28 | 25 |
| Matches the label in both orders | 19 | 23 | 23 |

The last column's 25 disagreements include 15 clear-preference questions and ten tie-labeled questions. The remaining twelve error questions are consistently wrong against the label in both orders. The other 23 questions match in both orders. We have improved total consistency from baseline, but still have a substantial position problem. Swapping during training was already implemented; adding it again is not a new fix.

All six original-order regressions from 20 to 200 updates are ties. Four become stable B preferences; two become A in original order while staying ties when swapped. Their topics include fashion jokes, PWA characteristics, backend-driven frontends, a Hamlet exchange, string comparisons, and a generic CEO role prompt. The content does not justify calling every lost tie a newly introduced factual error.

## Concrete examples

| Case ID ending | What is visible | What the 200-update judge does |
| --- | --- | --- |
| 9013: forest percentages | Both answers preserve the same nine supplied numbers and ordering; bullet styling differs. | Tie in original order, then prefers original B after it moves first. |
| 9016: five-day meal plan | A has 108 whitespace-separated words and all five day headings; B has 364 words against a 200-word limit and contains stray markup. This checks instructions, not nutrition. | Chooses whichever answer is first, including B despite the visible length failure. |
| 9019: nieces question | A omits Harry's seven children from a total; 2 + 7 + 2 = 11. B retains the missing-information issue without that omission. | Tie originally, then B when B moves first. |
| 8707: praline-scanning blog | A invents nutritional benefits; B invents over 1,000 training images and over 95% accuracy. Neither claim is in the user's supplied description. | Consistently prefers B; the publisher prefers A. Neither preference establishes factual support. |

On case **8844**, B's statement that Redis does not persist to disk is contradicted by the official [Redis persistence documentation](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/), which describes RDB and AOF. The expanded judge still prefers B when B is first. This check does not validate all of A's other claims.

On case **9077**, the user's conversation identifies Ophelia and Laertes as siblings, yet B repeats an expected-marriage claim. [Folger's edition, Act 1 scene 3](https://www.folger.edu/explore/shakespeares-works/hamlet/read/1/3/), confirms the sibling relationship. This narrow source check does not adjudicate every claim in the exchange or replace the publisher's tie label.

The [case-by-case review](ERROR_REVIEW_CASES.md) names the exact scope and uncertainty for each of the 37 questions. External facts were mostly left unchecked; no candidate code was executed. Six cases are flagged for future human adjudication. These flags do not remove them from the old scores or make validation examples eligible for training.

## Chosen next experiment: change one sentence, freeze the weights

**Test the tie definition before selecting another weight-training change.** A comparison that changes one factor is often called an **ablation**. Here, only one prompt sentence changes:

- Current: `Use tie_preference when their overall answer quality is equal.`
- Candidate: `Use tie_preference when their overall answer quality is about the same.`

Run the candidate prompt on the original Qwen model and the saved 200-update adapter: 60 questions × two orders × two models = **240 new decisions**. The existing old-prompt results provide each model's control. Keeping both models helps distinguish a prompt effect from an adapter-specific effect.

The planned limits are 900 seconds per model worker, 4,096 input tokens, 16 output tokens, temperature zero, no automatic retries, no paid APIs, and **zero training updates**. Keep the output parser, all answer labels, the sample, and the rest of the prompt unchanged. The old protected instruction file must remain untouched; use the candidate text saved in the separate plan.

For the 200-update adapter, the proposed development screen requires at least 7/20 original ties and 9/20 swapped ties, while retaining at least 28/40 and 27/40 clear-winner matches respectively, 35/60 consistent choices, and 23/60 both-order matches. All 240 decisions must be valid. Report totals, individual regressions, and the original model's response to the prompt change regardless of whether the screen passes. These exploratory thresholds are set before the proposed run; they are not evidence of statistical significance or public-release readiness.

If this fails, retain the result and reassess. If it passes, the next discussion can specify rebuilding **training-only** conversations with that instruction and a comparison using the same prompt. We have not selected a new learning rate, extra updates, class weighting, or validation-label edits. Increasing tie weighting could also increase false ties on clear winners, as the first trial already demonstrated a trade-off.

## Saved evidence and checks

- [60-question prediction table](results/error_review_20261006T221309230086Z/prediction_matrix.jsonl): all three versions, both orders, original labels, word counts, and released vote strengths.
- [37 diagnostic notes](results/error_review_20261006T221309230086Z/case_reviews.jsonl): model observations, uncertainty tags, and unchanged labels; all marked ineligible for training.
- [Count summary](results/error_review_20261006T221309230086Z/summary.json) and [two scoped source checks](results/error_review_20261006T221309230086Z/source_checks.json).
- [Exact proposed next experiment](results/error_review_20261006T221309230086Z/next_experiment_plan.json): candidate prompt, controls, limits, and success screen; status is `proposed_not_run`.
- [Verification receipt](results/error_review_20261006T221309230086Z/review_verification.json): input fingerprints, full coverage checks, and baseline protection.
- [Execution note](results/error_review_20261006T221309230086Z/execution_notes.json): an initial temporary-export parsing failure and the corrected newline reader. The input data and existing application reader were unchanged.

All 35 protected baseline files, 14 first-trial artifacts, and 52 expanded-trial artifacts passed their preservation checks. No application or test code changed. This advances judge development; it does not connect the judge to the product.
