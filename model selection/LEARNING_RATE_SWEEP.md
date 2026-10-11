# The monitored learning-rate comparison

**Later follow-up, October 7:** numerical corrections were made and Colab Pro was selected. [The corrected package](CORRECTED_JUDGE_COLAB.md) is prepared and locally checked. Re-sweep the rates after the CUDA probe; do not automatically reuse the historical winning rate for longer training. The measurements and decisions below describe the original completed sweep.

**Status, October 7, 2026: completed and audited.** All three rates finished; the monitor selected `2e-5` at update 150. Full validation reached **246/311 clear matches (79.1%)**, log-loss **0.454**, and AUC **0.878**. The probability-quality development screen passes; the accuracy screen does not. Read [the measured result and next decision](LEARNING_RATE_SWEEP_RESULTS.md). The 2,000-pair run is justified but has not started. Original Qwen JSON remains the baseline-v2 reference at **239/311 (76.8%)**.

A **learning rate** controls how far the optimizer moves learned values in one update. A **sweep** compares a few rates under the same conditions. Here we compare `1e-5`, `2e-5`, and `5e-5`: 0.00001, 0.00002, and 0.00005. The completed fixed-monitor comparison selected `2e-5` at update 150 for full validation.

## What happens after what?

1. **Save the rules first.** The [saved plan](results/judge_sweep_20261007T093531415353Z/plan.json) records the settings, data order, metrics, improvement thresholds, deadlines, and exact source/input fingerprints before the sweep. The original baselines and historical checkpoints stay protected.
2. **Reuse the selected task.** Keep the binary JSON prompt chosen under baseline v2. Reuse the prepared 2,000-pair A/B pool. For this short comparison, each rate sees the same first **600 pairs in 599 question groups**, both orders: **1,200 presentations**. A full pass over all 2,000 pairs has not run.
3. **Start each rate the same way.** Load the original Qwen weights and initialize the same rank-16 LoRA adapter, with scale 20 and dropout 0.05, on all 36 blocks and seven projections per block. LoRA learns additional matrices while the original model stays frozen. Only the peak learning rate differs between candidates.
4. **Train for 150 updates.** One update averages the learning signals from four pairs in both orders. Use the existing binary loss, the measured memory fix, 50 warm-up updates, and cosine decay to ten percent of the peak by update 150. There is no performance-based early stop in this short comparison; every complete candidate gets the same 150-update opportunity.
5. **Save complete progress every 25 updates.** A checkpoint stores adapter values, optimizer state, private random state, schedule position, and data cursor. The next segment starts in a fresh Python process and restores this checkpoint. The runner checks that every segment starts from its predecessor's exact saved state.
6. **Pause training to measure at updates 100 and 150.** Use the same **100 clear validation pairs in 100 groups**, each in both orders. Disable dropout while measuring; verify that adapter, optimizer, and random state remain unchanged; then restore training mode. These validation questions never become learning examples.
7. **Select from six complete measurements.** The three rates each contribute two measured checkpoints. Choose the lowest averaged-probability **log-loss**, a penalty for assigning too little probability to the preferred answer. Lower is better. Exact ties prefer the lower learning rate, then the earlier checkpoint. Missing, duplicated, non-finite, or incomplete measurements cannot produce a winner.
8. **Run full validation on the selected checkpoint.** Score all **396 pairs in both orders**, for 792 score readings. The main grade uses the **311 clear A/B pairs**; the 85 preference ties remain diagnostics. Do not pick a different checkpoint after inspecting this full-validation result.
9. **Compare with baseline v2 and save the result.** Report accuracy, log-loss, AUC, preference-strength and shorter-winner slices, order agreement, tie diagnostics, grouped 95% bootstrap intervals, and paired McNemar. The runner writes `results.json`, `RESULTS.md`, and `sweep_verification.json` inside the run folder automatically when all work succeeds.
10. **Use the result to plan the longer run.** Selecting the best rate does not automatically make it better than the untrained reference. The 2,000-pair training pass, later strength experiments, calibration, reserved final test, and product integration remain later steps.

## What has actually passed?

- **16 offline tests:** five new scheduling/selection/recovery checks, three existing training-data checks, and eight existing v2 metric checks.
- The existing synthetic GPU checks for binary loss, gradients, accumulation, dropout recomputation, schedule boundaries, and saved state.
- Two continuous rehearsal updates with validation inserted after each one, followed by a fresh-process replay of update two with the same validation calls.
- Exact adapter, optimizer, private random state, second-update losses/gradient norms, and monitor probabilities across the two routes. Evaluation also leaves numerical training state unchanged.

The [preflight receipt](results/judge_sweep_20261007T093531415353Z/probe_verification.json) records **three optimizer applications and sixteen diagnostic score readings**. The rehearsal uses two monitor pairs to test the machinery; it is not the full 100-pair monitor or a new judge-quality result. Peak rehearsal memory was **9.739 GiB**. The earlier [full-length memory check](TRAINING_MEMORY.md) passed at 10.216 GiB for both orders of a 2,045-token prompt under the same 18-GiB guideline.

## How much work was completed?

| Item | Fixed amount |
| --- | --- |
| Rates | 3 |
| Updates per rate | 150 |
| Training presentations per rate | 1,200 |
| Total sweep updates / presentations | 450 / 3,600 |
| Validation monitor | 100 pairs × 2 orders × 2 checkpoints × 3 rates = 1,200 scores |
| Selected-checkpoint full validation | 396 pairs × 2 orders = 792 scores |
| Checkpoint interval | Every 25 updates |
| Memory guideline | 18 GiB of MLX memory |
| Candidate wall-time budget | At most 4 hours per rate |
| Segment limit | 50 minutes, or 70 minutes when the segment includes a monitor |
| Selected full-validation limit | 1 hour |
| Sweep plus full-validation budget | At most 13 hours of bounded workers, excluding the short preflight and reporting |

The measured training speed suggests about **7.4 hours for the 3,600 training presentations**, plus validation and saving overhead: roughly **8–9 hours overall**. The actual sweep prompts average 894.7 tokens and reach 2,040 tokens. That was the pre-run estimate. Actual training/monitor and full-validation workers took **9.16 hours**, excluding preparation, preflight and reporting; full validation accounted for 37.83 minutes. The completed run stayed within its preregistered budgets. macOS `caffeinate` prevented idle sleep during the supervisor.

The runner stops on a worker failure, a deadline, invalid numerical values, a memory violation, or changed source/input files. It does not automatically retry. Completed segments can be reused; an explicit recovery command preserves an unsuccessful attempt before replaying from the preceding complete checkpoint. Previously spent worker time still counts against the budget. An abrupt termination may leave the runner lock behind; do not remove it until confirming no supervisor remains active.

## What counts as improvement?

The plan keeps accuracy on clear labels as the main number and sets two development screens in advance:

- Accuracy gain of at least **3 percentage points**, a paired 95% accuracy-change interval above zero, exact McNemar p < 0.05, and no increase in log-loss; or
- Log-loss reduction of at least **0.02**, a paired 95% loss-change interval below zero, and no observed accuracy decrease.

The checkpoint was selected using a subset of the same validation data. Full-validation intervals and tests therefore describe **exploratory development evidence after selection**, not an independent confirmation. Calibration and final-test contents remain unopened. These measurements concern agreement with human preferences, not proof that an answer is factually correct.

## Files and commands

Use [sweep_judge.py](sweep_judge.py) for the new workflow. It calls the unchanged [train_judge.py](train_judge.py) and v2 scoring/statistics helpers.

From the project root, read the completed report with:

```sh
cat "model selection/results/judge_sweep_20261007T093531415353Z/RESULTS.md"
```

The completed run used `run`, wrapped in `caffeinate -i`. Preserve its sealed folder; do not start another supervisor on it. Running `prepare` creates a separate plan and does not launch training. A completed sweep is sealed and cannot be rerun in place.

This work develops the local judge. The product still uses saved demo answers and quotation rules. Foundation-model pretraining remains separate.
