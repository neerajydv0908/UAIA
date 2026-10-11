# Baseline v2: full-validation results

**Completed October 6, 2026, local time (October 7 UTC).** All four model/format runs completed: **3,168 valid score reads**, covering all **396 validation pairs in both orders** for each version. No new training occurred. Baseline v1 and both saved adapters remained unchanged. Calibration and final-test contents remained unopened.

**Keep JSON as the measured reference.** The plain-text formatting bundle did not pass the selection rule saved before inference.

This is a development comparison of agreement with human preferences. It does not measure factual truth or source support, and it does not qualify a model for product integration by itself.

## 1. What happened, step by step?

1. Saved the [protocol](BASELINE_V2_PROTOCOL.md) and [machine-readable plan](results/baseline_v2_20261007T004058039152Z/plan.json) before inference, including all selection thresholds.
2. Used all 396 validation pairs: **311 clear A/B labels** for the main grade, with **85 ties** kept as a separate diagnostic. The pairs span 395 question groups; the clear pairs span 310.
3. Asked each fixed model to choose A or B. Read the raw next-token scores at that answer position, without generating a textual choice.
4. Converted the A/B scores into a probability, reversed the displayed answers, scored again, mapped both results to the same original answer, and averaged them.
5. Compared original Qwen, the saved 20-update adapter, the saved 200-update adapter, and original Qwen with the plain-text formatting bundle. Also measured two simple rules on the same labels.
6. Calculated uncertainty, paired comparisons and diagnostic slices. Independently checked the saved records and historical fingerprints, then applied the rule chosen before scoring.

An **adapter** is the small set of previously trained additions to the original model. **Inference** means using its existing numbers; **training** would change those numbers. This milestone performed inference only. The product demo still uses saved answers and quotation checks; this separate experiment made real local model calculations.

## 2. Main result: clear-winner preference accuracy

Every model row uses the averaged probability and all 311 clear pairs. Higher accuracy and AUC are better; lower log-loss is better. Brackets contain 95% intervals from 10,000 resamples of whole question groups.

| Version | Clear matches | Accuracy [95% interval] ↑ | Log-loss [95% interval] ↓ | AUC [95% interval] ↑ |
| --- | ---: | --- | --- | --- |
| Original Qwen · JSON (baseline v2) | 239/311 | 76.8% [72.0%, 81.5%] | 1.160 [0.848, 1.507] | 0.806 [0.755, 0.853] |
| 20-update adapter · JSON | 238/311 | 76.5% [71.7%, 81.2%] | 0.611 [0.510, 0.725] | 0.811 [0.761, 0.858] |
| 200-update adapter · JSON | 234/311 | 75.2% [70.4%, 80.0%] | 0.536 [0.488, 0.586] | 0.816 [0.768, 0.863] |
| Original Qwen · plain text | 236/311 | 75.9% [71.0%, 80.6%] | 1.266 [0.919, 1.652] | 0.814 [0.764, 0.861] |
| Always choose original B | 171/311 | 55.0% [49.5%, 60.6%] | Unavailable: hard rule | 0.500 [0.500, 0.500] |
| Choose the longer answer | 192/311 | 61.7% [56.3%, 67.1%] | Unavailable: hard rule | 0.614 [0.558, 0.669] |

The existing adapters changed clear matches from 239 to 238 after 20 updates and 234 after 200 updates. Their log-loss changed from 1.160 to 0.611 and 0.536. Read these as separate outcomes: a better probability score does not automatically mean more correct winner selections. The paired comparisons below show the uncertainty in each change.

**Accuracy** asks how often the selected answer matches the saved A/B label. Exactly 50/50 is counted as too close to call and therefore does not match a clear label. **Log-loss** penalizes confident mistakes. **AUC** measures how well the score ranks A-labelled examples above B-labelled examples; a constant score gives 0.5.

The rules have no estimated probabilities, so log-loss is unavailable. Their AUC uses hard scores and is a limited ranking reference. Always-B means the original dataset answer B, not whichever answer is displayed second after swapping.

For a mathematical reference, assigning exactly 50/50 to every clear pair gives log-loss `ln(2) = 0.693`. A model can have good accuracy and worse log-loss if its mistakes are too confident. This is why we retain all three metrics and postpone confidence calibration until a model is final.

These are new binary-protocol scores. Do not compare them directly with v1's three-class scores on 60 balanced questions. Both old adapters were trained under the earlier three-choice task; re-scoring them here is a useful historical reference, not a test of the proposed binary-training recipe.

## 3. Did scoring both orders help?

“Swapped” below means the result has already been mapped back to the original answer identity. Each accuracy denominator is 311. Order agreement uses all 396 pairs, including ties; the probability gap is the mean absolute difference between the two mapped probabilities.

| Version | Original matches | Swapped matches | Averaged matches | Orders agree | Mean probability gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Qwen · JSON (baseline v2) | 213/311 | 216/311 | 239/311 | 232/396 (58.6%) | 0.405 |
| 20-update adapter · JSON | 207/311 | 210/311 | 238/311 | 170/396 (42.9%) | 0.517 |
| 200-update adapter · JSON | 214/311 | 206/311 | 234/311 | 202/396 (51.0%) | 0.285 |
| Original Qwen · plain text | 217/311 | 221/311 | 236/311 | 254/396 (64.1%) | 0.352 |

For original Qwen in JSON, averaging raised accuracy from **68.5% to 76.8%**: **+8.4 percentage points**, with an exploratory paired 95% interval of **+4.2 to +12.8 points**. All four versions had higher observed accuracy after averaging than in either single order. Their full paired intervals remain in the saved metrics.

Averaging makes the combined score symmetric when answer identities are exchanged. It does not prove every bias is gone. A judge that always selects the first displayed answer can average to an uninformative 50/50. The saved metrics include exploratory paired intervals for averaging versus original order, separate from the three primary comparisons.

## 4. Are stronger preferences and shorter winners easier?

All slice numbers below use averaged probabilities. Strength 1 is the weakest released preference and strength 3 the strongest. “Shorter” means fewer Unicode characters, fixed before scoring. Equal-length answers are excluded from that slice.

| Version | Strength 1 (139) | Strength 2 (98) | Strength 3 (74) | Shorter winner (119) |
| --- | ---: | ---: | ---: | ---: |
| Original Qwen · JSON (baseline v2) | 103/139 (74.1%) | 76/98 (77.6%) | 60/74 (81.1%) | 71/119 (59.7%) |
| 20-update adapter · JSON | 102/139 (73.4%) | 76/98 (77.6%) | 60/74 (81.1%) | 71/119 (59.7%) |
| 200-update adapter · JSON | 99/139 (71.2%) | 75/98 (76.5%) | 60/74 (81.1%) | 70/119 (58.8%) |
| Original Qwen · plain text | 99/139 (71.2%) | 76/98 (77.6%) | 61/74 (82.4%) | 77/119 (64.7%) |
| Always choose original B | 78/139 (56.1%) | 52/98 (53.1%) | 41/74 (55.4%) | 61/119 (51.3%) |
| Choose the longer answer | 88/139 (63.3%) | 61/98 (62.2%) | 43/74 (58.1%) | 0/119 (0.0%) |

The [complete metrics](results/baseline_v2_20261007T004058039152Z/metrics.json) include accuracy, log-loss and AUC intervals for every slice. These smaller slices are exploratory and have wider uncertainty; a visible pattern alone does not establish its cause.

## 5. What happened on the 85 ties?

An overall mean near 0.5 can hide confident choices in opposite directions. The mean absolute distance from 0.5 shows whether individual pairs are actually close. The “45–55%” column counts pairs whose averaged probability lies within 0.05 of 0.5.

| Version | Mean P(original A) | Mean distance from 0.5 | Within 45–55% | Selected margin | Tie labels caught | Clear pairs kept as A/B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Qwen · JSON (baseline v2) | 0.443 | 0.210 | 37/85 | ±0.000 | 0/85 | 311/311 |
| 20-update adapter · JSON | 0.466 | 0.127 | 44/85 | ±0.000 | 3/85 | 311/311 |
| 200-update adapter · JSON | 0.475 | 0.119 | 31/85 | ±0.000 | 4/85 | 306/311 |
| Original Qwen · plain text | 0.439 | 0.262 | 31/85 | ±0.000 | 0/85 | 310/311 |

Margins were selected from the preregistered grid while retaining at least 95% of clear examples as A/B decisions. A selected margin is a **validation-fitted diagnostic**, not a calibrated or final product threshold. It means the judge is too close to choose reliably; it does not establish that both answers are correct. The headline scores above remain separate from this margin experiment.

A selected width of zero means no positive tested width met the coverage requirement. It does not establish useful tie detection; the table must be read together with the number of tie labels caught.

All four versions selected zero width. The 200-update adapter had five clear-labelled pairs at exactly 50/50; those count as non-matches under the predefined accuracy rule. Plain text had one such clear pair. These outcomes are retained rather than resolved by an arbitrary A/B tie-break.

All inspected training/validation aggregate ties contained opposing reviewer votes. Future binary training will exclude those tie-labelled pairs without changing their labels or claiming that disagreement is meaningless. The old 29-tie content-review step is dropped.

## 6. Paired comparisons and the format decision

Changes are candidate minus baseline-v2 JSON, using the same 311 pairs. Accuracy changes are percentage points (pp); negative log-loss changes are improvements. McNemar compares corrections against regressions using one predefined representative per clear question group (310 groups). Holm adjustment accounts for the three primary tests.

| Candidate | Accuracy change [95% interval] | Log-loss change [95% interval] | Corrections / regressions (310 groups) | Exact p | Holm-adjusted p |
| --- | --- | --- | ---: | ---: | ---: |
| 20-update adapter · JSON | -0.3 pp [-1.0, +0.0] | -0.549 [-0.791, -0.328] | 0 / 1 | 1.0000 | 1.0000 |
| 200-update adapter · JSON | -1.6 pp [-3.5, +0.0] | -0.624 [-0.929, -0.349] | 1 / 6 | 0.1250 | 0.3750 |
| Original Qwen · plain text | -1.0 pp [-4.5, +2.6] | +0.107 [-0.137, +0.343] | 14 / 16 | 0.8555 | 1.0000 |

Across all 311 clear pairs, plain text corrected 14 decisions and regressed on 17. The McNemar table has 16 regressions because its predefined one-per-question rule removes the repeated question's additional pair from that test only. Both pairs remain in headline scores and grouped intervals.

The accuracy route required at least **+3 percentage points**, a paired interval entirely above zero, adjusted p below 0.05, and no log-loss increase. The single format comparison had one alternative: log-loss improvement of at least **0.02**, a paired interval entirely below zero, and no observed accuracy decline. The probability route does not claim a statistically significant accuracy gain.

**Keep JSON as the measured reference.** The plain-text formatting bundle did not pass the selection rule saved before inference.

This does not establish that JSON is universally better: the paired intervals include zero change. Plain text improved observed order agreement and shorter-winner accuracy, while its main accuracy and log-loss did not improve. Those diagnostic gains do not override the criteria fixed before the run.

The formatting candidate changed a bundle: readable sections and real line breaks, readable NVIDIA conversation markers in the question, and omission of the empty evidence field. The answer text and binary instruction were held fixed. Any measured effect belongs to that bundle; this experiment cannot isolate which component caused it.

## 7. Were execution and verification successful?

| Run | Score reads | Worker minutes | Mean seconds/read | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: |
| Original Qwen · JSON (baseline v2) | 792/792 | 27.89 | 2.10 | 8.12 GiB |
| 20-update adapter · JSON | 792/792 | 27.90 | 2.11 | 8.14 GiB |
| 200-update adapter · JSON | 792/792 | 27.12 | 2.05 | 8.14 GiB |
| Original Qwen · plain text | 792/792 | 32.32 | 2.44 | 8.12 GiB |

The four workers took **115.24 minutes** in total. The plain-text run slowed during its later reads and finished slower overall despite shorter prompts. These are observed sequential-run timings, not a controlled speed benchmark.

- All 3,168 score reads completed with no missing records or damaged tails. No validation input was truncated; maximum lengths were 2,429 JSON tokens and 2,314 plain-text tokens, under the 4,096-token cap.
- Eight offline tests passed before inference. The short direct-versus-cached numerical probe passed; it is an implementation check, not a judge-quality score.
- The independent audit checked every score identity and probability, 112 metric point estimates, 16 grouped accuracy/log-loss intervals, and all three primary paired tests. AUC point estimates were separately checked by comparing every A-labelled score with every B-labelled score; AUC intervals use the tested rank-based implementation.
- The 35 protected baseline files and the 14, 52, 6 and 12 artifacts from the earlier milestones matched their existing receipts. Both loaded adapters matched their original trained numerical fingerprints before and after inference.
- All 34 installed package versions still matched the prepared environment. The audit also checked all 28 tie-band candidates and independently recomputed the format decision.
- Full historical fingerprint checks ran once before and once after the milestone.
- Zero new training updates, paid API calls or model downloads. The 775-example calibration and 769-example final-test contents remained unopened.

Execution success does not mean a candidate passed the quality criterion. Those are separate results.

## 8. What do the probabilities leave out?

A/B probabilities are normalized over **only A and B**. They are not calibrated confidence in truth. The raw records also preserve full-vocabulary probabilities, allowing us to check whether the model put substantial mass on other possible tokens.

| Version | Mean full-vocabulary mass on A/B | Reads with A/B mass below 50% | Top token outside A/B |
| --- | ---: | ---: | ---: |
| Original Qwen · JSON (baseline v2) | 99.97% | 0/792 | 0/792 |
| 20-update adapter · JSON | 99.90% | 1/792 | 1/792 |
| 200-update adapter · JSON | 99.40% | 1/792 | 1/792 |
| Original Qwen · plain text | >99.99% | 0/792 | 0/792 |

The saved `tie` score is for token 48731 only. The historical `tie_preference` label requires two tokens, so the prefix probability must not be reported as a full-label probability. These token-mass diagnostics were calculated after the run and do not change the preregistered format criterion.

The intervals describe this validation sample under grouped resampling. Repeated development can still overfit validation. Both answer orders are dependent readings of one pair; they do not double the independent sample size. The final held-out test remains reserved for the locked judge.

## 9. What happens next?

The next milestone is a **bounded binary-training readiness probe**, followed by the preplanned learning-rate sweep if the probe passes. Use the selected **JSON** template consistently in training and evaluation. Create a new `train_judge.py`; preserve the historical training runners. Check the all-36-layer rank-16 LoRA setup, eight-example gradient accumulation, paired orientations, two-way A/B loss, scheduling, memory, speed and exact checkpoint resumption before an overnight run.

The [revised roadmap](JUDGE_ROADMAP_V2.md) records the later strength experiment, 2,000-pair run, full eligible-data run, final-model calibration, one-time final test, evidence data and product integration. Those stages have **not run** in this milestone. The optional Skywork reference remains deferred because HelpSteer2 overlap has not been ruled out and its separate PyTorch runtime is not installed.

HelpSteer2 teaches preference; it cannot by itself teach evidence checking.

## Saved evidence

[Plan](results/baseline_v2_20261007T004058039152Z/plan.json) · [complete metrics](results/baseline_v2_20261007T004058039152Z/metrics.json) · [execution](results/baseline_v2_20261007T004058039152Z/execution.json) · [independent audit](results/baseline_v2_20261007T004058039152Z/independent_audit.json) · [artifact verification](results/baseline_v2_20261007T004058039152Z/v2_verification.json) · [comparable experiment table](EXPERIMENTS_V2.md).
