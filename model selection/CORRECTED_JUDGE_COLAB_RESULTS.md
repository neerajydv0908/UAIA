# Corrected Colab judge: measured results

Completed October 7, 2026 at 22:54 UTC. **All four planned steps are complete. A longer 2,000-pair experiment is justified by the predefined probability-quality rule; it has not been launched.**

The selected judge matched **252/311 clear preferences (81.0%)**, compared with **238/311 (76.5%)** for unchanged Qwen under the same CUDA/FP32 scoring. Log-loss fell from **1.169 to 0.450**. Log-loss measures probability error and penalizes confident mistakes; lower is better. AUC measures how well scores rank A-preferred versus B-preferred pairs; higher is better.

## Full validation

Both models scored all **396 pairs in both orders**, giving 792 score reads each. Map the second order back to the original answers, then average the two probabilities. Main accuracy uses the 311 clear labels; all 85 aggregate ties remain separate diagnostics. There are 310 independent clear-question groups. Intervals below use 10,000 resamples of whole question groups with seed 20261007.

| Version | Clear matches | Accuracy [95% interval] ↑ | Log-loss [95% interval] ↓ | AUC [95% interval] ↑ |
| --- | ---: | --- | --- | --- |
| Original Qwen · JSON · CUDA/FP32 baseline | 238/311 | 76.5% [71.7%, 81.2%] | 1.169 [0.855, 1.521] | 0.804 [0.754, 0.853] |
| Corrected LoRA · JSON · CUDA/FP32 · 2e-5, update 150 | 252/311 | 81.0% [76.6%, 85.2%] | 0.450 [0.398, 0.502] | 0.871 [0.830, 0.910] |

Historical **baseline v2 stays 239/311 (76.8%)**, with its MLX/BF16 output scoring unchanged. Its selected Mac adapter stays 246/311 (79.1%). These new CUDA/FP32 rows do not overwrite those records; backend and recipe differences prevent attributing their difference to one correction alone.

## Paired comparison and decision

A paired comparison checks both models on the same questions. The saved plan's thresholds were not changed after seeing results.

| Selected minus unchanged | Observed change | Paired 95% interval |
| --- | ---: | --- |
| Accuracy | +4.50 percentage points | +0.32 to +8.68 points |
| Log-loss | −0.719 | −1.048 to −0.429 |
| AUC | +0.0667 | +0.0289 to +0.1057 |

Thirty clear decisions improved and sixteen regressed. The exact two-sided **McNemar p-value is 0.05408**. McNemar tests whether one model wins more often among disagreements. It uses one predetermined representative per question group; the accuracy and bootstrap retain all pairs. This is one planned primary comparison, so the adjusted p-value is the same.

- **Accuracy route: does not pass.** The gain exceeds three points and its bootstrap interval is above zero, but p = 0.05408 misses the required p < 0.05. Different uncertainty methods can disagree near a cutoff; retain the preregistered requirement.
- **Probability route: passes.** Log-loss decreases by more than 0.02, its entire paired interval is below zero, and observed accuracy does not fall.
- **Decision: prepare a bounded 2,000-pair experiment using corrected rate 2e-5.** Start a fresh adapter and a schedule appropriate to the longer run, retain the independent monitor, and preregister its budget before launch. The present authorization ends with this decision. No longer run was started.

## How the checkpoint was selected

Each candidate saw the same first 600 clear pairs, in both orders, over 150 updates. Selection used 200 held-out training-side pairs outside the 2,000-pair pool, with no validation-group overlap. All six readings completed before choosing the lowest monitor log-loss.

| Peak learning rate | Update | Monitor matches | Monitor log-loss |
| --- | ---: | ---: | ---: |
| 1e-05 | 100 | 145/200 | 0.620857980 |
| 1e-05 | 150 | 144/200 | 0.531421948 |
| 2e-05 | 100 | 141/200 | 0.726706610 |
| 2e-05 | 150 | 145/200 | 0.530685544 |
| 5e-05 | 100 | 139/200 | 0.766889937 |
| 5e-05 | 150 | 142/200 | 0.536601036 |

Selected: `sweep/lr_2e-05/checkpoint_0150`. The runner compared unrounded values, retaining its lower-rate/earlier-step tiebreak rule. The runner did not use full-validation results to select this checkpoint. The close monitor values do not establish that 2e-5 is universally superior.

Fixed recipe: Qwen3-4B-Instruct-2507 with frozen BF16 base weights; float32 A/B projection before softmax; rank-16 LoRA on seven projections in all 36 layers, alpha 320 and dropout 0.05; global accumulated-gradient clipping at 1.0; bias-corrected Adam; four pairs/eight presentations per update; 50-update warmup then cosine decay; seed 20261007; hard labels, JSON and no length weighting. Source, bundle and environment fingerprints matched the passing CUDA probe before and after validation. Soft labels, smoothing, length weighting, readable formatting and top-16-layer changes were not part of this experiment.

## Diagnostic slices

| Diagnostic | Unchanged Qwen | Selected judge |
| --- | ---: | ---: |
| Preference strength 1 | 103/139 (74.1%) | 112/139 (80.6%) |
| Preference strength 2 | 75/98 (76.5%) | 76/98 (77.6%) |
| Preference strength 3 | 60/74 (81.1%) | 64/74 (86.5%) |
| Shorter answer won | 71/119 (59.7%) | 81/119 (68.1%) |
| Single-turn question | 175/221 (79.2%) | 183/221 (82.8%) |
| Multi-turn question | 63/90 (70.0%) | 69/90 (76.7%) |
| Both answer orders agree (all 396 pairs) | 235/396 (59.3%) | 320/396 (80.8%) |

The strongest preferences have the best selected accuracy, but strength 2 is below strength 1; improvement is not monotonic. Shorter-winner accuracy improves by 8.4 points, while remaining well below overall accuracy. Its individual 95% intervals are 51.3–68.9% before and 59.7–76.5% after; this slice has no separate preregistered significance claim. Multi-turn questions also improve but remain harder. Both-order averaging does not prove that all position effects have disappeared.

Before averaging, original-order clear matches were 214/311 for unchanged Qwen and 247/311 for selected; swapped-order matches were 216/311 and 247/311. Two-order averaging raises these to 238 and 252 respectively. The exact reports retain each order's log-loss, AUC and intervals.

## Ties and confidence

All **85 tie-labelled pairs** are retained. Their mean probability for original A changes from 0.4457 to 0.4540; mean absolute distance from 0.5 changes only from **0.2069 to 0.2052**. Tie probabilities therefore are not reliably close to 50/50.

At an illustrative ±0.05 band, the share of tie labels inside the band falls from **39/85 (45.9%) to 9/85 (10.6%)**, while clear-pair coverage rises from 71.4% to 90.4%. At ±0.10, the selected judge puts 28/85 ties inside the band and retains 83.9% of clear pairs. These are diagnostic bands, not fitted product thresholds. Aggregate ties can reflect differing vote shares; the hard-label run did not train on them.

Coverage means the fraction of clear pairs retained after setting aside the smallest confidence margins. The following points are descriptive and use validation again, not calibration.

| Requested coverage | Actual retained | Actual coverage | Unchanged accuracy | Selected accuracy |
| --- | ---: | ---: | ---: | ---: |
| 100% | 311/311 | 100.0% | 76.5% | 81.0% |
| 90% | 280/311 | 90.0% | 76.8% | 84.6% |
| 75% | 233/311 | 74.9% | 79.8% | 86.7% |
| 50% | 156/311 | 50.2% | 84.0% | 89.1% |

The selected judge reaches **86.7% accuracy on its most confident 74.9%** of clear pairs. This does not establish a deployable abstention threshold or calibrated correctness probability. No calibration fitting or final-test model evaluation has run.

## Timing and evidence

The three training workers took 81.14 minutes in total, including their monitors, with 9.114 GiB peak allocated CUDA memory. Baseline validation took 245.45 seconds and selected validation 255.65 seconds. Both finished within their one-hour ceilings. The sequential driver started at 22:45:41 UTC and completed at 22:54:02 UTC. Every worker exited successfully; no retry occurred.

The [completion audit](results/colab_validation_20261007T224541Z/completion_audit.json) checked all 450 update steps, 2,400 monitor score reads and 1,584 full-validation reads. It recomputed both complete reports, coverage, question-group intervals and the paired screen from saved scores. Every numerical report field matched within 1e-11. All 92 sweep-export files and 26 full-validation-export files matched their exported SHA-256 fingerprints. The selected checkpoint was checked during its successful restore; large model and optimizer files remain on Drive rather than being downloaded locally.

- [Full evidence and helper guides](results/colab_validation_20261007T224541Z/README.md).
- [Unchanged model report](results/colab_validation_20261007T224541Z/full_evidence/baseline_fp32/full_report.json).
- [Selected model report](results/colab_validation_20261007T224541Z/full_evidence/selected_fp32/full_report.json).
- [Post-run source and bundle verification](results/colab_validation_20261007T224541Z/full_evidence/post_validation_verification.json).
- [Preregistered recipe and screen](CORRECTED_JUDGE_COLAB.md).

Drive folder: `MyDrive/judge_corrected_20261007`. Original logs, checkpoints, optimizer states and failure/setup history are preserved.

## Limits on the conclusion

These are **exploratory development measurements**, from one seed, one dataset and repeated use of validation. Moving this checkpoint-selection monitor to training-side questions improves separation but cannot erase earlier development use of validation. The experiment supports trying more data, not declaring a final judge or product improvement. Evidence verification itself is not taught by preference-only HelpSteer2 data.

Some calibration text was incidentally exposed during an earlier broad text search, so the reserved split files should not be described as entirely uninspected. No calibration fitting or final-test model scoring was performed, and split data was not changed. Product integration remains pending. Longer training, calibrated abstention and final-test performance require later milestones.

The notebook displays the completed results. Colab subsequently disconnected for inactivity after reports were saved; no worker failed and no reconnect or rerun was needed.
