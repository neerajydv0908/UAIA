# Comparable judge experiments: protocol v2

Updated October 7, 2026. **The corrected CUDA/FP32 comparison is complete.** Selected accuracy is 81.0% versus 76.5% for its matching unchanged baseline. The probability-quality screen passes; a longer 2,000-pair experiment is justified but not launched. Historical MLX/BF16-head rows remain preserved. [Full measured report](CORRECTED_JUDGE_COLAB_RESULTS.md).

 Every row below uses all 311 clear A/B labels from the same 396-pair validation set and the same grouped 95% interval method. Model rows average both mapped answer orders. Ties are diagnosed separately. Keep future experiments in this table only when they follow this protocol; mark deviations explicitly.

| Version | Clear matches | Accuracy [95% interval] ↑ | Log-loss [95% interval] ↓ | AUC [95% interval] ↑ |
| --- | ---: | --- | --- | --- |
| Original Qwen · JSON (baseline v2) | 239/311 | 76.8% [72.0%, 81.5%] | 1.160 [0.848, 1.507] | 0.806 [0.755, 0.853] |
| 20-update adapter · JSON | 238/311 | 76.5% [71.7%, 81.2%] | 0.611 [0.510, 0.725] | 0.811 [0.761, 0.858] |
| 200-update adapter · JSON | 234/311 | 75.2% [70.4%, 80.0%] | 0.536 [0.488, 0.586] | 0.816 [0.768, 0.863] |
| Original Qwen · plain text | 236/311 | 75.9% [71.0%, 80.6%] | 1.266 [0.919, 1.652] | 0.814 [0.764, 0.861] |
| Binary sweep · JSON · 2e-5 at update 150 (selected on validation) | 246/311 | 79.1% [74.5%, 83.5%] | 0.454 [0.411, 0.498] | 0.878 [0.839, 0.915] |
| Original Qwen · JSON · CUDA/FP32 baseline | 238/311 | 76.5% [71.7%, 81.2%] | 1.169 [0.855, 1.521] | 0.804 [0.754, 0.853] |
| Corrected LoRA · JSON · CUDA/FP32 · 2e-5, update 150 | 252/311 | 81.0% [76.6%, 85.2%] | 0.450 [0.398, 0.502] | 0.871 [0.830, 0.910] |
| Always choose original B | 171/311 | 55.0% [49.5%, 60.6%] | Unavailable: hard rule | 0.500 [0.500, 0.500] |
| Choose the longer answer | 192/311 | 61.7% [56.3%, 67.1%] | Unavailable: hard rule | 0.614 [0.558, 0.669] |

**Experiment:** inference-only measurement and formatting comparison. **Hypothesis:** readable sections improve original Qwen's probability scoring; two-order averaging may improve decisions. **Changes:** the plain-text bundle is compared with JSON using unchanged original weights and the same binary rubric; the historical adapters are separate reference versions. **Minimum gain:** +3 accuracy points with the additional paired-test requirements, or the predefined 0.02 log-loss route for the single format contrast. **Decision:** retain **JSON** for the next training stage. See [the full result](BASELINE_V2.md) for both orders, slices, ties, paired tests, limits and runtime.

Source: [baseline_v2_20261007T004058039152Z](results/baseline_v2_20261007T004058039152Z/metrics.json). Baseline v1 is [locked history](BASELINE.md); its 60-question, three-class numbers are not comparable rows here. The binary sweep is complete; no full-pool training, calibration, or final-test result is available yet. Future rows must name their saved plan, data/format version, training seed, model checkpoint and changed component.

The [binary-training readiness milestone](BINARY_TRAINING_READINESS.md) is an engineering check, so it adds no judge-quality row to this table. The original full-length memory check failed. The [follow-up memory fix](TRAINING_MEMORY.md) passed at 10.216 GiB with exact full-probe resumption. These engineering updates add no full-validation score or promoted adapter.

The [monitored learning-rate sweep](LEARNING_RATE_SWEEP_RESULTS.md) completed under [its preregistered plan](results/judge_sweep_20261007T093531415353Z/plan.json). **Hypothesis:** broader binary LoRA improves probability quality. **Changed component:** peak learning rate (`1e-5`, `2e-5`, `5e-5`), with identical initialization, seed 20261007, JSON input, and 600 training pairs in both orders per candidate. **Selected checkpoint:** `lr_2e-5_0150/checkpoint_0150`, Qwen3-4B-Instruct-2507 with frozen original weights and rank-16 LoRA on all 36 blocks. The unchanged 2,000-pair prepared pool supplies its first 600 pairs; only those 600 were used per trial.

**Decision:** probability-quality development screen passes: log-loss change −0.705 [−1.037, −0.406], with no observed accuracy decline. Accuracy change +2.25 points [−1.94, +6.45], McNemar p = 0.371, fails the accuracy route. That milestone justified a larger experiment. The subsequent numerical corrections require a new rate sweep first; no 2,000-pair run has started. Shorter-winner accuracy fell from 59.7% to 56.3%, and reliable tie handling remains unresolved. Because selection used part of the same validation set, this row and its intervals are exploratory development evidence, not independent confirmation. See [saved full results](results/judge_sweep_20261007T093531415353Z/results.json) and [post-run audit](results/judge_sweep_review_20261007/audit.json).

## Historical accuracy versus coverage — descriptive, uncalibrated

Coverage means the fraction of clear pairs retained when lower-confidence decisions are set aside. Confidence here is distance from 0.5 after averaging both orders. Requested fractions are rounded to the nearest pair; all pairs tied at the boundary are kept together. These points reuse validation and the validation-selected adapter. They are not a fitted deployment threshold, calibrated probabilities, or independent confirmation.

| Requested coverage | Actual retained | Actual coverage | Clear accuracy |
| --- | ---: | ---: | ---: |
| 100% | 311/311 | 100.0% | 79.1% |
| 90% | 280/311 | 90.0% | 81.8% |
| 75% | 233/311 | 74.9% | 87.1% |
| 50% | 156/311 | 50.2% | 91.0% |

Source: [saved correction audit](results/colab_preparation_20261007/correction_audit.json), derived from the sealed selected adapter's paired scores. Temperature fitting and choosing an abstention threshold remain reserved for calibration after the model is locked.

## Corrected CUDA experiment — complete

**Hypothesis:** the corrected recipe improves probability quality relative to unchanged Qwen scored with the same CUDA/FP32 head. **Sweep change:** peak rate only (1e-5, 2e-5, 5e-5); identical seed 20261007, 600 clear training pairs in both orders, hard labels, JSON, all 36 layers, rank 16, clipping 1.0 and bias-corrected Adam. All six 200-pair training-side monitor readings completed; minimum monitor log-loss selected **2e-5 at update 150**. This monitor is outside the 2,000-pair training pool and validation groups. All 450 updates and 1,584 full-validation score reads passed.

**Unchanged screen:** accuracy gain ≥3 points, paired interval above zero, McNemar p < .05, and no log-loss rise; alternatively log-loss decrease ≥.02 with its paired interval below zero and no observed accuracy decline. **Result:** probability route passes, accuracy route fails. Accuracy delta **+4.50 points [0.32, 8.68]**, McNemar **p = .05408**; log-loss delta **−.719 [−1.048, −.429]**; AUC delta **+.0667 [.0289, .1057]**. Thirty pairs improve, sixteen regress. The decision supports a bounded longer experiment, not an independently confirmed accuracy claim.

Shorter-winner accuracy is **59.7% → 68.1%**; order agreement **59.3% → 80.8%**. Strength 1/2/3 selected accuracies are **80.6% / 77.6% / 86.5%**. All 85 ties remain diagnostic: mean absolute distance from .5 is **.2069 → .2052**, and only 9/85 selected tie labels fall within ±.05. Tie handling is unresolved. Soft labels, length weighting and trained-format contrasts have not run.

The [full results](CORRECTED_JUDGE_COLAB_RESULTS.md), [selected report](results/colab_validation_20261007T224541Z/full_evidence/selected_fp32/full_report.json) and [recomputed audit](results/colab_validation_20261007T224541Z/completion_audit.json) preserve the plan, source/data identities and all diagnostics. Moving checkpoint selection off validation does not erase earlier development use; the reported intervals remain exploratory. No calibration fitting or final-test model evaluation has run. The earlier incidental split-content exposure still applies.

### Corrected accuracy versus coverage — descriptive, uncalibrated

Confidence ranks clear pairs by their averaged distance from .5. Keep boundary ties together. These validation-derived points do not set a deployment threshold.

| Requested coverage | Actual retained | Actual coverage | Unchanged accuracy | Selected accuracy |
| --- | ---: | ---: | ---: | ---: |
| 100% | 311/311 | 100.0% | 76.5% | 81.0% |
| 90% | 280/311 | 90.0% | 76.8% | 84.6% |
| 75% | 233/311 | 74.9% | 79.8% | 86.7% |
| 50% | 156/311 | 50.2% | 84.0% | 89.1% |

Decision: **prepare the longer 2,000-pair experiment; not launched.** Calibration and choosing a final abstention threshold remain later work.

## Superseded data-curve proposal — not executed

The [data curve](DATA_CURVE_PLAN.md) fixes 1,200 / 2,400 / 4,646 clear pairs, a versioned 400-pair training-side monitor, rate 2e-5 and one full pass each. Strength weighting and second-pass shuffling are implemented and checked; the curve uses unweighted hard labels. Budget and new CUDA readiness remain pending. Historical metrics above are unchanged; the monitor-only 75.0% ensemble is not a validation-table entry.

## Current experiment — full-data A100, running

[Fixed run plan](A100_FULL_DATA.md): one fresh corrected Qwen adapter, all 4,646 clear pairs for one pass (1,162 updates), 400 training-side monitor pairs, rate 2e-5/seed 20261007, no loss/format/model changes. Monitor/checkpoint interval 100, then the final update; complete the full pass before selecting the lowest monitor log-loss. Retain the final checkpoint and the selected version's actual exposure. Six-hour compute ceiling; no retries. This supersedes the unexecuted size curve.

Hypothesis: the full-pool corrected judge improves decisions or probability quality versus unchanged Qwen on the same A100. Primary full-validation comparison uses the existing +3-point accuracy route or .02 log-loss route, including the paired interval and McNemar conditions recorded above. Report all slices, order agreement and coverage. Historical L4 scores are context, not a new controlled contrast with the 600-pair adapter. **No result row or new accuracy claim until both evaluations complete.** Seventeen local and Colab tests and package checks pass. The A100 40 GB full-model probe passed; the single training worker is running. Both full evaluations are pending, with no new result row yet. [Observed live status](results/a100_full_preparation_20261007/cloud_status.json).
