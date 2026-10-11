# Expanded trial: 100 groups, 200 updates

October 6, 2026. **Completed: 200 updates, exact reproduction of the earlier update-20 adapter, a passed fresh-process reload, and all 120 validation decisions. The larger trial is a mixed result; it remains experimental.**

This is the next experiment after the [20-update trial](TRAINING_TRIAL.md). It teaches the same small additions to Qwen, called **LoRA adapters**, using all 100 prepared training question groups in both answer orders. One **update** changes those adapter numbers using one presented example. The 200 presentations contain 100 distinct questions, not 200 independent questions.

## What we measured

| Measure | Protected baseline | After 20 updates | After 200 updates |
| --- | ---: | ---: | ---: |
| Original-order preference agreement | 30/60 — 50.0% | 36/60 — 60.0% | 34/60 — 56.7% |
| Swapped-order preference agreement | 31/60 — 51.7% | 34/60 — 56.7% | 35/60 — 58.3% |
| Expected A, original order | 16/20 | 13/20 | 15/20 |
| Expected B, original order | 12/20 | 11/20 | 13/20 |
| Expected tie, original order | 2/20 | 12/20 | 6/20 |
| Clear A/B winner labels combined | 28/40 | 24/40 | 28/40 |
| Same underlying choice after swapping | 30/60 | 32/60 | 35/60 |
| Expected label matched in both orders | 19/60 | 23/60 | 23/60 |
| Invalid / failed / skipped / missing decisions | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Compared with the 20-update version, the larger trial gained four clear-winner matches but lost six tie matches in the original order. The total therefore fell by two matches. It predicted ten original-order ties: six on expected ties and four on clear preferences. The earlier trial predicted 22 ties, twelve on expected ties and ten on clear preferences. This is an observed trade-off, not proof of its cause or future performance.

Compared with baseline, five original-order questions improved and one regressed. Compared with the 20-update trial, four improved and six regressed. For swapped order, the respective counts were four improved/zero regressed and six improved/five regressed. No comparison was unresolved. Individual record IDs are saved for review.

The expanded adapter still changes its underlying choice on **25 of 60 questions** when answer positions are reversed. Both-order matches remain 23/60, the same as the earlier adapter. More practice did not produce a consistent improvement across these measures. The unchanged longer-answer rule matches 26/60 original labels; constant-A, constant-B, and constant-tie rules each match 20/60.

## Decision and next move

**Keep the baseline and both adapters. Do not promote this adapter into the product.** The training path works, and the larger experiment gives us useful evidence about remaining problems. These repeated development results do not establish a reliable preference judge or a source-checking judge.

Next review the saved examples behind the six original-order regressions from the 20-update version, the missed ties, and the 25 answer-order disagreements. Then choose one justified change, record its update budget and comparison rules before running it, and preserve these three references. This review and the next training change remain future work; the current milestone stops after the completed comparison and artifact audit. Calibration and final-test records remain reserved for their intended later roles.

## What happens after what?

1. **Check our references.** Verify the protected original model and baseline, the earlier trial's artifacts, and the prepared training files.
2. **Save the plan.** Fix the 200-presentation order before training. Keep the model, learning settings, and validation rules unchanged.
3. **Start from the original model.** Freeze its original weights. Create the same small adapter and a fresh optimizer, the calculation that applies learning updates.
4. **Reproduce the first 20 updates.** Use exactly the earlier trial's ten groups, answer orders, and initialization seed. A seed makes the random choices repeatable.
5. **Apply 180 more updates.** Show each of the remaining 90 training groups twice, reversing A and B on the second presentation. Correctly reverse the A/B teaching label too; a tie stays a tie.
6. **Save separate checkpoints.** Save adapter numbers every ten updates. A checkpoint is a saved version we can inspect or load. The final update-200 checkpoint is selected in advance for validation.
7. **Check the reproduction.** Compare the saved update-20 adapter's numerical fingerprint with the earlier trial. Stop before validation if they differ. This check occurs after the training worker finishes.
8. **Reload in a fresh process.** Load the original model plus the final adapter. Require exact adapter values and matching checked next-token scores, called logits.
9. **Compare all three versions.** Use the same 60 validation question groups, each in both answer orders. Compare the original baseline, the 20-update trial, and this 200-update trial. Report ties, clear A/B preferences, and answer-order stability separately.
10. **Check preservation again.** Confirm the baseline and earlier run are unchanged, then record the result and next decision.

The ordinary product demo still reads saved answers and checks quotations. This experiment makes real local model calculations and changes separate adapter weights. It is fine-tuning an existing model, not training one from scratch.

## Fixed settings and limits

| Setting | This experiment |
| --- | --- |
| Original model | Qwen3-4B-Instruct-2507, original BF16, same downloaded revision |
| Trainable additions | Rank-8 LoRA, scale 20, zero dropout, q/v projections in the last four blocks; 327,680 numbers |
| Training examples | The same prepared 100 training groups, each in both answer orders, used once per orientation |
| Teaching labels across 200 presentations | 71 A, 71 B, 58 ties |
| Learning | Adam optimizer, learning rate 0.00001, batch size 1, seed 20261006 |
| Loss | Learn from the assistant label and its conversation ending; exclude prompt and padding tokens using the previously checked correction |
| Input length | At most 2,048 training tokens; no truncation |
| Memory | Checkpoint recomputation enabled; 18 GiB MLX guideline, not a hard whole-Mac memory cap |
| Bound | Exactly 200 planned updates, training worker at most 1,200 seconds, reload/evaluation worker at most 900 seconds |
| Validation | Fixed 60 groups, 120 planned decisions, same prompt, temperature zero, 4,096 input tokens and 16 output tokens maximum |
| Recovery | Separate adapter checkpoints only; no saved optimizer state or automatic resumption |
| Spending and reserved data | Local files and Mac GPU; no paid APIs, downloads, calibration records, or final-test records |

The saved readiness configuration still contains the earlier 20-update proposal. It is preserved as historical input. The new plan's schedule, `planned_updates`, `training_wall_seconds`, and `evaluation_wall_seconds` record this experiment's actual bounds.

We start fresh because the earlier adapter did not save optimizer state. Loading that adapter with a fresh optimizer would change the learning path. Reproducing the original 20 updates in one continuous run lets us extend that path and check its reproduction explicitly. This version still cannot resume an interrupted training run exactly.

## What the comparison can tell us

The labels describe human answer preference in HelpSteer2. They do not independently establish factual truth or source support. Validation is balanced at 20 A, 20 B, and 20 tie labels; its percentage does not estimate ordinary user traffic.

Both presentations of a question belong to the same group. Swapped choices are translated back to the original answer identities before comparison. Invalid, failed, skipped, and missing results remain distinct from passing judgments. The unchanged simple-rule comparators remain available in the saved metrics.

These same validation questions have now guided development repeatedly. The result is a development comparison, not an untouched final test or evidence of a production-ready judge. Intermediate checkpoints are retained but are not evaluated to choose the best-looking one. Known group overlap is excluded; earlier audit limitations and unknown public-data exposure during Qwen pretraining still apply.

## Training and reload checks

Training plus the update-20 reproduction check took **605.04 seconds**, about ten minutes five seconds. All 200 updates passed the finite-number, target-count, optimizer-step, and memory checks. The run used exactly 100 groups and every prepared presentation once. Actual sequences ranged from **174 to 1,931 tokens**, without truncation. There were **658 taught tokens**, including label text and conversation endings. Peak training MLX allocation was **9.32 GiB**; this is a library allocation reading, not total Mac memory.

The first twenty logged example IDs, answer orders, labels, sequence lengths, taught-token counts, and losses exactly match the earlier trial. Its saved update-20 adapter also has the same numerical fingerprint. The final adapter contains **327,680 trainable numbers** and occupies **1,312,445 bytes**, about 1.31 MB, separate from the original model. Its numbers changed through training, and fresh-process reload recovered them exactly. The checked logits had **zero maximum absolute difference**; the permitted absolute and relative tolerance was 0.00001.

Reload and all 120 validation decisions took **256.09 seconds**, about four minutes sixteen seconds. Median decision time was **1.91 seconds**, p95 **3.12 seconds**, and peak evaluation MLX allocation **8.42 GiB**. The two workers together took about fourteen minutes twenty-one seconds, excluding supervisor checks. Small timing differences between runs do not establish a speed improvement.

A separate saved-result audit checked the full schedule, original/swapped label pairing, all 200 log records, the exact twenty-update prefix, source/preparation fingerprints, known train/validation group separation, 120 strict output parses, prompt token lengths against baseline, category and paired-change counts, and reload records. It saved fingerprints for all **52 run artifacts**. The final baseline guard passed for all **35 protected files**, and all **14 earlier trial artifacts** matched their receipt. No earlier checkpoint or baseline file was overwritten.

## Saved artifacts

The new run is in `model selection/results/expanded_trial_20261006T132929116227Z`.

| Artifact | Purpose |
| --- | --- |
| [Plan](results/expanded_trial_20261006T132929116227Z/plan.json) | Exact 200-update schedule, unchanged learning settings, actual bounds, and file fingerprints |
| [Training log](results/expanded_trial_20261006T132929116227Z/training_log.jsonl) | Every actual update, label, loss, time, and memory reading |
| [Training result](results/expanded_trial_20261006T132929116227Z/training_result.json) | Changed adapter fingerprint, update count, and save information |
| [Update-20 reproduction](results/expanded_trial_20261006T132929116227Z/prefix_check.json) | Exact numerical match with the first experiment |
| [Final adapter](results/expanded_trial_20261006T132929116227Z/adapter/adapters.safetensors) | Separate update-200 learned additions |
| [Adapter settings](results/expanded_trial_20261006T132929116227Z/adapter/adapter_config.json) | Structure needed to load the additions alongside the original model |
| [Reload check](results/expanded_trial_20261006T132929116227Z/reload_check.json) | Exact recovered values and matching forward scores |
| [Raw decisions](results/expanded_trial_20261006T132929116227Z/evaluation_predictions.jsonl) | All 120 actual validation outputs and their outcomes |
| [Three-version comparison](results/expanded_trial_20261006T132929116227Z/comparison.json) | Metrics, worker statuses, paired changes, and preservation checks |
| [Error lists](results/expanded_trial_20261006T132929116227Z/error_analysis.json) | IDs of improvements, regressions, position disagreements, and ties on clear labels |
| [Independent audit](results/expanded_trial_20261006T132929116227Z/expanded_verification.json) | Verification checks and fingerprints for all other run files |

Checkpoints at updates 10, 20, …, 190 are also retained. Update 20 was checked for reproduction; none of the intermediate checkpoints was evaluated to choose the highest validation score. These local artifacts remain outside ordinary Git commits. Earlier reports describe their own historical milestones and remain available unchanged.

## Run and read the code

From the project folder, with Mac GPU access:

```bash
"model selection/.venv/bin/python" "model selection/expand_trial.py"
```

This command starts a new real training experiment in a new timestamped folder. It preserves the earlier scripts and artifacts. The new launcher reuses their unchanged training and evaluation functions.

The four new offline tests use fictional records and no GPU:

```bash
"model selection/.venv/bin/python" -m unittest discover -s "model selection" -p test_expand_trial.py -v
```

All four passed. They check the exact earlier schedule prefix, complete paired coverage, repeatability, invalid schedules, changed earlier artifacts, and comparison counts that retain regressions and missing results.
