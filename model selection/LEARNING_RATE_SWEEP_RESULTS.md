# Learning-rate sweep: measured result and next decision

**Later follow-up, October 7:** numerical corrections were made and Colab Pro was selected. [The corrected package](CORRECTED_JUDGE_COLAB.md) is prepared and locally checked. Re-sweep the rates after the CUDA probe; do not automatically reuse the historical winning rate for longer training. The measurements and decisions below describe the original completed sweep.

**Completed October 7, 2026. Decision: the evidence supports a bounded 2,000-pair training experiment using peak learning rate `2e-5`. That longer run has not started.** The current four-step request is complete: train the three candidates, select one by the fixed monitor, evaluate all validation pairs, and decide what to do next.

## What happened, step by step

1. Trained three identically initialized adapters for 150 updates each. Every candidate saw the same 600 A/B training pairs in both orders, with eight presentations per update. The original Qwen weights stayed frozen.
2. Measured each candidate at updates 100 and 150 on the same separate 100-pair validation monitor.
3. Selected `2e-5` at update 150 because it had the lowest monitor log-loss. The selection was saved before full validation.
4. Evaluated the selected checkpoint on all 396 validation pairs in both orders: 792 score readings. Accuracy uses the 311 clear labels; 85 ties remain separate diagnostics.
5. Verified the saved artifacts and recomputed the probabilities, full metrics, grouped intervals, paired tests, selection, and development screen from saved outputs. No new inference or training was needed for this review.

## Comparable full-validation results

| Model | Clear matches | Accuracy [95% interval] | Log-loss [95% interval] | AUC [95% interval] |
| --- | ---: | --- | --- | --- |
| Original Qwen JSON (baseline v2) | 239/311 | 76.8% [72.0%, 81.5%] | 1.160 [0.848, 1.507] | 0.806 [0.755, 0.853] |
| Selected binary adapter, 2e-5 at update 150 | 246/311 | 79.1% [74.5%, 83.5%] | 0.454 [0.411, 0.498] | 0.878 [0.839, 0.915] |

**Log-loss** penalizes assigning low probability to the preferred answer; lower is better. **AUC** measures how well A/B probabilities rank the label groups; higher is better. Neither metric measures factual verification or calibrated confidence.

The accuracy change is **+2.25 percentage points**, with a paired 95% interval of **−1.94 to +6.45 points**. There were 26 corrected decisions and 19 regressions. The paired McNemar test, which compares these disagreements by question group, gives **p = 0.371**. The improvement is not statistically established under the preregistered rule.

Log-loss changed by **−0.705** (paired interval **−1.037 to −0.406**). AUC changed by **+0.073** (paired interval **+0.031 to +0.115**).

## All six monitor measurements

These use only the fixed 100-pair monitor and must not be confused with the 311-pair main grade.

| Peak rate | Update | Monitor accuracy | Monitor log-loss | Monitor AUC |
| --- | ---: | ---: | ---: | ---: |
| 1e-5 | 100 | 80.0% | 0.465137 | 0.861 |
| 1e-5 | 150 | 84.0% | 0.457523 | 0.874 |
| 2e-5 | 100 | 80.0% | 0.447783 | 0.876 |
| 2e-5 | 150 | 83.0% | 0.429637 | 0.896 |
| 5e-5 | 100 | 74.0% | 0.630940 | 0.777 |
| 5e-5 | 150 | 72.0% | 0.560254 | 0.793 |

## Diagnostic checks

| Slice | Baseline v2 | Selected adapter |
| --- | ---: | ---: |
| Preference strength 1 | 103/139 (74.1%) | 104/139 (74.8%) |
| Preference strength 2 | 76/98 (77.6%) | 78/98 (79.6%) |
| Preference strength 3 | 60/74 (81.1%) | 64/74 (86.5%) |
| Shorter answer was preferred | 71/119 (59.7%) | 67/119 (56.3%) |

- Accuracy increases with preference strength in the selected adapter: 74.8%, 79.6%, and 86.5%.
- Shorter-winner accuracy falls from **71/119 (59.7%) to 67/119 (56.3%)**. This is a remaining weakness to monitor during the next controlled experiment, not proof that the model only measures length.
- Answer-order agreement improves from **232/396 (58.6%) to 299/396 (75.5%)**. Mean probability disagreement falls from 0.405 to 0.131. Residual order sensitivity remains.
- Original-order accuracy is 237/311; swapped-order accuracy is 239/311; averaging reaches 246/311. The averaged view remains the predefined primary result.
- On the 85 tie-labelled pairs, mean P(original A) is **0.445**, and mean absolute distance from 0.5 is **0.185**, versus 0.210 for baseline. An average near 0.5 can hide confident individual choices, so the distance is also reported.
- The declared tie-band rule still selects width **0**. A 0.025 band retains 96.8% clear coverage but has a lower balanced tie-detection score than width 0; wider tested bands miss the 95% coverage constraint. Reliable tie handling is not established. The zero-width diagnostic catches 3/85 ties and abstains on 4/311 clear pairs.

## Decision under the rules written before training

**Accuracy route: does not pass.** The observed gain is below the 3-point target, its paired interval includes zero, and McNemar p exceeds 0.05.

**Probability route: passes.** The log-loss reduction exceeds 0.02, its paired interval is entirely below zero, and observed accuracy did not decline. This supports the next development experiment, rather than deployment.

**Next experiment direction:** use `2e-5` as the peak-rate candidate for a bounded run on the prepared 2,000-pair pool. Before launching, save its new schedule, checkpoint/monitor cadence, early-stop rule, and runtime budget. A single pass would contain 4,000 presentations and 500 eight-presentation updates. Start from the original model with a newly initialized adapter under that new plan; do not append updates to the sweep checkpoint with its already-completed 150-update decay schedule. Keep strength weighting as a separate later comparison. Longer training is justified to investigate; improvement is not guaranteed.

The 100-pair selection monitor overlaps full validation. These intervals and tests are **exploratory development evidence after selection**, not an independent confirmation. Calibration and the reserved final test remain unopened; no product integration has occurred. This preference-only dataset still cannot teach evidence checking.

## Execution and verification

All 18 training segments completed without recorded failures or retries: **450 updates / 3,600 presentations**, **1,200 monitor scores**, and **792 full-validation scores**. Peak training memory was **10.331 GiB**, below the 18-GiB guideline. Training-segment workers including their monitors took 8.53 hours; full validation took 37.83 minutes. Combined bounded-worker time was **9.16 hours**, excluding preparation, preflight and report overhead; this is slightly longer than the original rough 8–9-hour estimate.

The post-run review verified **314 sealed artifact fingerprints**, unchanged active sources/inputs, identical candidate initialization, exact continuity between segments, and unchanged state during all seven evaluations. The runner had already checked historical receipts before and after training. The existing 16 offline tests and GPU preflight remain the preparation evidence; source code was not changed or retested during this documentation-only completion review.

Sources: [saved plan](results/judge_sweep_20261007T093531415353Z/plan.json), [selection](results/judge_sweep_20261007T093531415353Z/selection.json), [complete metrics](results/judge_sweep_20261007T093531415353Z/results.json), [sealed execution receipt](results/judge_sweep_20261007T093531415353Z/sweep_verification.json), and [post-run audit](results/judge_sweep_review_20261007/audit.json). The sealed run files remain unchanged.
