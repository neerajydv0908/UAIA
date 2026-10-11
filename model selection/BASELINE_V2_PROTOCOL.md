# Baseline v2: measure before training again

Preregistered October 6, 2026, local time; the saved run folder uses October 7 UTC. This document and the [saved plan](results/baseline_v2_20261007T004058039152Z/plan.json) define the experiment before any v2 model scores are read. Results belong in [BASELINE_V2.md](BASELINE_V2.md). Baseline v1 and its original experiments stay unchanged.

## What happens after what?

1. Check the existing baseline and experiment receipts once at the start of this milestone.
2. Read all validation pairs and verify their saved file fingerprint. Read training metadata to check separation and counts; do not open calibration or final-test examples.
3. Fix the A/B task, input formats, token identities, metrics, statistical rules, and run limits before inference.
4. Score the original Qwen, the 20-update adapter, and the 200-update adapter using the same JSON format and instruction. Then score original Qwen with the proposed plain-text format. Each worker handles all 396 pairs in both orders.
5. Map both probabilities to the original answer identity and average them. Measure both individual orders as well, so the effect of averaging is visible.
6. Recompute scores, slices, uncertainty, and paired comparisons from the saved probabilities. Compare simple rules too.
7. Choose the input format using the rule below. Fit a provisional tie margin on validation, keeping its diagnostic status explicit.
8. Recheck historical files and new artifacts once at the end. No new training happens in this phase.

## What was checked locally?

- Validation: **396 answer pairs, 395 question groups, 311 A/B labels, 85 ties**. Two A-labelled pairs share a question group; neither is discarded from headline scores.
- Clear preferences by strength: 139 at strength 1, 98 at strength 2, and 74 at strength 3.
- Training: 6,623 pairs, including 5,147 A/B pairs. The historical 2,048-token eligibility check admitted **5,037 A/B pairs**: 2,360 A and 2,677 B. Eligibility must be rechecked before training with another format; this is not a universal data-count limit.
- All 1,476 training ties and all 85 validation ties have opposing released individual votes, with no individual zero vote. This describes the inspected splits, not unopened calibration/test records or proof that all ties are meaningless.
- NVIDIA conversation markers occur in 106/396 validation questions, about 27%, and 1,814/6,623 training questions. All inspected validation evidence lists are empty.
- The local tokenizer encodes `A` as `[32]`, `B` as `[33]`, `tie` as `[48731]`, and `tie_preference` as **two tokens**, `[48731, 92718]`. Complete chat-template checks confirm that A/B begin at the scored decision position in both formats and both orders.
- Existing release metadata lists 775 calibration and 769 final-test examples. Their contents remain unopened.

The publisher recommends its dedicated preference annotations for Bradley–Terry modelling. That supports investigating a pairwise objective; it does not establish that every aggregate tie is noise. Our immediate decision is to train future binary experiments on A/B examples, retaining ties for diagnostics. [NVIDIA dataset card](https://huggingface.co/datasets/nvidia/HelpSteer2#preferences-new---1-oct-2024).

## Exactly what reaches the model?

All four jobs use the protected original rubric with the output request changed to **A or B** and the tie instruction removed. The final sentence says that a preference alone does not certify factual correctness. Baseline v2 therefore defines a new binary evaluation protocol; its score is not directly comparable to v1's three-class score on 60 balanced questions.

The JSON control preserves the existing question/evidence/answer serialization. The candidate is one predefined formatting bundle:

```text
### Question
The supplied question, with User: and Assistant: turns where needed

### Answer A
The first candidate, with real line breaks

### Answer B
The second candidate, with real line breaks
```

The candidate rewrites only `<extra_id_1>User` and `<extra_id_1>Assistant` markers in the question, trims surrounding question whitespace, and omits the empty evidence field. Answer text is unchanged apart from position swapping. Labels, strength, reviewer explanations, and split metadata never enter the prompt. The original instruction still identifies supplied content as data, not instructions changing the judging rules. Any measured effect belongs to the **formatting bundle**, not to an individually isolated formatting change.

## What do the probability scores mean?

A **logit** is the model's raw score for a possible next token. At the first answer position, save the A, B, and tie-prefix logits, their full-vocabulary log probabilities, and their three-token normalized scores. The main probability is:

```text
p_display_A = exp(logit_A) / (exp(logit_A) + exp(logit_B))
p_original_A = p_display_A on the original input
p_swapped_A  = 1 - p_display_A on the reversed input
p_averaged_A = (p_original_A + p_swapped_A) / 2
```

The code subtracts the largest logit before exponentiation to avoid numerical overflow. These are probabilities **conditional on choosing A or B**, not calibrated probabilities of factual correctness. The tie-prefix score is not the probability of completing the two-token `tie_preference` label. Full-vocabulary scores and the top token remain available to detect probability mass outside the requested choices.

The model does not generate a full textual answer. Chunked prompt processing saves memory; one short synthetic probe compares it with a direct forward calculation before the first worker scores validation pairs. We allow at most 0.25 raw-logit difference across the vocabulary and 0.01 A/B-probability difference for that BF16 numerical check; actual differences are saved. This check does not prove the quality of any model judgment.

Two-order averaging makes the combined probability symmetric to exchanging answer identities, assuming the same deterministic scores are reused. It does **not** prove all bias is gone. A model that always favors the displayed first answer can average to an uninformative 50/50. Measure the disagreement and probability gap before averaging as well as the combined result. Exactly 0.5 is a too-close-to-call decision and counts as a non-match on clear labels.

## Scores and uncertainty

The main number is **A/B preference accuracy on all 311 clear pairs**, using the averaged probability. Also report accuracy for each individual order. This means matching the saved preference label, not independently verifying truth.

- **Log-loss:** penalizes probabilities that confidently favor the wrong answer; lower is better. Use natural logarithms and clip only for calculation to `[1e-12, 1-1e-12]`.
- **AUC:** how well the score ranks A-labelled versus B-labelled examples; 0.5 is a constant-score reference, and higher is better. Equal scores get half credit in ranking.
- **Slices:** strengths 1, 2, 3, and the 119 clear pairs where the shorter answer won. Length means Unicode character count, fixed before scoring; equal lengths are excluded from this slice.
- **Tie diagnostics:** average distance from 50/50, counts within fixed margin bands, and the tie-prefix score. Ties never contribute one third of the headline grade.
- **Position diagnostics:** agreement of the original-identity decisions across orders and the mean absolute probability gap.
- **Simple rules:** always choose original B and pick the longer answer. Their accuracy and hard-score AUC use the same clear examples and group intervals. A hard rule supplies no estimated probability, so its log-loss is explicitly unavailable instead of inventing confidence. Always-B is a canonical-label reference; no order-consistency claim is made for it.

Use **10,000 bootstrap resamples**, seed 20261007, taking whole question groups with replacement. All rows of a repeated question move together; both orientations have already been combined into one pair record. Report the 2.5th and 97.5th percentiles as the 95% interval for accuracy, log-loss, and AUC. Comparisons reuse the same resampled groups for both versions. Slice intervals are exploratory and pointwise. This follows paired resampling principles; the project implements the grouped extension directly using NumPy. [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

**McNemar's test** compares paired correctness changes. Its exact version uses the binomial distribution on gains versus losses. To avoid treating the repeated question as independent twice, use the lexically first clear-pair ID per question group for that test: 310 independent question representatives. Headline accuracy and the grouped bootstrap still retain all 311 clear pairs. Save all-pair gains/losses separately. Apply Holm's adjustment to the three predeclared model/format comparisons. Averaging-versus-single-order comparisons are exploratory and clearly labelled. [Statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

The ±5-point estimate is approximate, not a promise: the worst-case normal half-width on 311 independent binary outcomes is about 5.6 points, and actual grouped intervals depend on the results. Two orders do not double the independent sample size. Repeated use of validation can overfit development choices; the final test remains reserved.

Missing, invalid, or interrupted score reads remain unresolved. They prevent a complete result and format adoption. No subset of successful calls silently replaces the planned denominator.

## Criteria fixed before the run

The three primary comparisons are the 20-update adapter versus original Qwen, the 200-update adapter versus original Qwen, and plain-text original Qwen versus JSON original Qwen. Use the same averaged A/B metrics throughout.

A practical accuracy gain is **at least 3 percentage points**, roughly ten additional matches among 311 pairs. For accepting the plain format through accuracy, also require a positive lower bound on the paired accuracy interval, Holm-adjusted exact McNemar p below 0.05, and no increase in log-loss.

There is one predeclared alternative for the single format-selection contrast: **log-loss falls by at least 0.02**, the entire paired 95% log-loss-change interval is below zero, and observed accuracy does not fall. This chooses a probability improvement for development; it is not a statistically significant accuracy-gain claim. If neither path passes, keep JSON as the measured reference. A failed or incomplete worker prevents format adoption.

For diagnostic ties, try only half-widths **0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20** around 0.5. Among bands retaining at least 95% of clear pairs as A/B decisions, maximize the mean of tie recall and clear coverage; break equal scores toward the smaller band. Save every candidate band. This validation-fitted margin means the judge cannot choose reliably between these candidates, not that both are correct. It is separate from primary forced A/B accuracy and from later probability calibration.

## Bounds and remaining stages

Four workers each read 792 decision-position scores: **3,168 planned reads**, plus two short synthetic probe score evaluations. Each worker has a 2,400-second wall deadline. Input cap is 4,096 tokens, with no truncation, no automatic retries, no paid calls, no downloads, and no training. The first two adapter files are loaded exactly and never updated. Runtime estimates are hypotheses until measured.

Full historical fingerprint checks occur once before and once after this milestone; small input checks remain inside workers. The larger roadmap is in [JUDGE_ROADMAP_V2.md](JUDGE_ROADMAP_V2.md). Calibration, test use, further training, external reward-model installation, and product integration are later stages.
