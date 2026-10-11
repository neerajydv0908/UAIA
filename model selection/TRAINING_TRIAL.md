# First real training trial: 20 small updates
Yes! let;s go

October 6, 2026. **The first LoRA training trial has run, its separate adapter has been saved, and a fresh-process reload check passed.** The fixed-sample comparison is complete: original-order preference agreement increased from 30/60 to 36/60, with better tie recognition but worse agreement on clear A/B winners.

The original model and baseline remain protected. This is our first experimental fine-tuning of an existing model.

## Before and after on the same questions

| Measure | Protected baseline | After 20 updates |
| --- | ---: | ---: |
| Original-order preference agreement | 30/60 — 50.0% | 36/60 — 60.0% |
| Swapped-order preference agreement | 31/60 — 51.7% | 34/60 — 56.7% |
| Expected A, original order | 16/20 | 13/20 |
| Expected B, original order | 12/20 | 11/20 |
| Expected tie, original order | 2/20 | 12/20 |
| Clear A/B winner labels combined | 28/40 | 24/40 |
| Same underlying choice after swapping | 30/60 | 32/60 |
| Label matched in both orders | 19/60 | 23/60 |
| Invalid / failed / skipped / missing evaluation decisions | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The original-order gain is **six additional matches, or ten percentage points**. Ten questions changed from a label mismatch to a match, while four changed from a match to a mismatch. In the swapped order, eight improved and five regressed. These are paired comparisons with the saved baseline outputs; the original model was not rerun or replaced.

Most of the gain came from ties. The trial predicted 22 original-order ties: 12 matched tie labels and ten occurred on examples labeled with a clear A/B preference. The baseline predicted four ties, two matching tie labels and two on clear-winner labels. The new adapter is therefore better at recognizing some ties but also more willing to call clear preferences ties. We have measured the trade-off; we have not established a general cause or a reliable improvement on new questions.

The longer-answer comparison rule still matches 26/60 original labels, and each constant-label rule still matches 20/60. All comparisons use the same small balanced development sample. No individual-answer confidence percentage is being claimed.

Reload plus all 120 evaluation decisions took **258.92 seconds**, about four minutes 19 seconds. Median evaluation time was **1.93 seconds**, p95 **3.11 seconds**, and peak evaluation MLX allocation **8.42 GiB**. Different runs can vary with warm-up and other computer activity; these small timing differences do not establish a speed improvement. Training and evaluation workers together took about five minutes 25 seconds, excluding supervisor checks.

The final guard passed for all 35 protected files. Calibration and final-test records were not opened. There was no automatic second training attempt and no prompt or label change to improve these scores.

## Decision and next move

**Keep this adapter as a reproducible experimental checkpoint. Continue using the original product's quotation rules.** Training, checkpoint saving, and fresh-process reload now work; product readiness is not established.

The next useful experiment is a larger controlled pass over the already-prepared training pool, with a fixed update budget and the same comparison protocol. A proposed next size is all 100 prepared groups in both orders, 200 presentations, rather than the ten groups used here. Choose and record its settings before seeing its validation results, and preserve both this trial and baseline v1. This larger run has not started.

Track tie recognition together with matches on clear A/B winners, answer-order stability, and invalid outputs. Do not accept a higher total score by itself. Before longer or interruptible training, add and verify optimizer-state saving if exact resumption is needed. A later source-grounded judge still requires reviewed evidence data, and a locked later model still needs the reserved final test.

## What actually changed?

Qwen already contained learned numbers, called **weights**, when we downloaded it. We froze those original weights and added small **LoRA adapters**: extra arrays of numbers beside selected calculations in the last four model blocks. A block is a repeated group of model calculations.

This trial trained an adapter containing **327,680 adjustable numbers** through 20 optimizer updates. The **optimizer**, Adam here, applies changes suggested by training examples. The originals on disk were neither overwritten nor merged with the new additions.

The saved adapter is 1,312,445 bytes, about 1.31 MB. It is not a standalone language model: to use it, load the original Qwen model and then these additions.

## What happened, step by step?

1. **Check the reference.** Verify the fingerprints and locks of all 35 protected baseline files. Also verify the prepared training files against the earlier readiness report.
2. **Choose a small fixed schedule.** From the prepared pool of 100 training groups, use the fixed seed to choose ten groups. Show both answer orders for each group, making 20 presentations and 20 updates. Alternate which orientation is shown first.
3. **Save the plan.** Record the exact schedule, settings, code fingerprints, source-data fingerprints, time limits, and the fixed baseline sample before changing any numbers.
4. **Load and freeze Qwen.** Attach the previously tested rank-8 LoRA arrays to query/value attention projections in the last four blocks. Check that only the same 16 adapter arrays are trainable.
5. **Teach one presentation at a time.** The model reads the question and two answers. Its expected A, B, or tie decision is the training target, not a hint inside the question.
6. **Calculate the learning error.** The **loss** measures how poorly the model predicts the target tokens. **Gradients** describe proposed changes to reduce that loss. Use the corrected answer-only loss from the readiness probe, excluding prompt and padding tokens.
7. **Check and update.** Require finite numbers, the exact expected target-token count, and memory within the guideline. Apply one Adam update to the adapter and verify the optimizer's step count.
8. **Save separately.** Save adapter checkpoints after updates 10 and 20. Preserve a log of every completed update. The final scheduled checkpoint is chosen in advance, not selected by trying several checkpoints on validation.
9. **Reload in a fresh process.** Load the protected base model and final adapter again. Compare every adapter value and the model's next-token scores on one training input with the pre-save values.
10. **Compare with the baseline.** Use its exact 60 validation questions in both orders, original judging instructions, token limits, strict parser, and temperature-zero generation. Save all new results in a separate folder and recheck baseline protection afterward.

The 100-group pool was prepared earlier; **only ten groups were used for weight updates in this trial**. Their 20 presentations contain eight A targets, eight B targets, and four tie targets. The complete training sequences ranged from 331 to 1,579 tokens. Altogether, the corrected loss taught 64 target tokens, including conversation endings.

## Fixed training settings

| Setting | Value |
| --- | --- |
| Starting model | Original BF16 Qwen3-4B-Instruct-2507, fixed downloaded revision |
| Learning method | LoRA rank 8, scale 20, dropout 0; q/v projections in the last four blocks |
| Trainable arrays / numbers | 16 / 327,680 |
| Optimizer / learning rate | Adam / 0.00001 |
| Batch size | One presentation per update |
| Update count | 20, with no automatic retry |
| Training input cap | 2,048 tokens including the target; no truncation |
| Training target | Only the assistant decision and conversation ending, excluding input and padding |
| Seed | 20261006 for group scheduling and adapter initialization |
| Gradient checkpointing | Enabled; recomputes intermediate calculations to reduce memory use |
| Worker deadline | 900 seconds for training, then a separate 900 seconds for reload and evaluation |
| Memory guideline | 18 GiB of MLX allocation, checked around updates; not a hard whole-Mac memory cap |
| Evaluation | Fixed 60 validation groups, both answer orders, at most 16 generated tokens and 4,096 input tokens |
| Paid APIs / Colab | None |

The exact settings inherited from readiness remain in the saved plan. Its historical `future_trial_status: proposed_not_run` describes the preparation snapshot; the actual worker and training-result files record this completed trial.

## Saving and reloading worked

The training worker completed in **66.08 seconds** and peaked at **9.15 GiB of MLX allocation**. This is library-reported peak memory, not the total memory used by the Mac.

The adapter's before/after fingerprints differ, confirming that training changed its numbers. The fresh process recovered exactly the same final adapter numbers. Its checked next-token scores, called **logits**, had **zero maximum absolute difference** from the pre-save reference. The permitted comparison tolerance was 0.00001 absolute and relative; the observed difference was zero.

This verifies this adapter checkpoint and one forward-output check. It does not prove the judge is accurate on arbitrary questions. The checkpoint contains adapter weights for inference; optimizer state is not saved, so this version does not support exact mid-run training resumption.

## How to interpret the comparison

We measure agreement with the public dataset's human preference labels. This is not factual-truth accuracy. The sample is deliberately balanced: 20 A labels, 20 B labels, and 20 ties. Its aggregate percentage does not estimate real user traffic.

Each validation question appears twice, so there are 120 decisions but only 60 independent question groups. A swapped prediction is translated back to the original answer identity before checking consistency. Extra text, truncated generation, errors, and missing decisions cannot count as a successful judgment.

Training loss on one example cannot be compared directly with loss on a different example as proof that quality improved. We use the unchanged validation sample for the before/after comparison. This is development data; a later tuned version still needs its separate final test. Public-data exposure in Qwen's earlier pretraining is unknown.

Twenty updates establish a working training path. Any observed score change is a small preliminary result, not evidence that this is a production-ready judge or that the full product is better than one good model. We do not alter the baseline or deploy this adapter into the application during this milestone.

## Saved artifacts

The run is in `model selection/results/training_trial_20261006T104025828291Z`.

| Artifact | Purpose |
| --- | --- |
| [Plan](results/training_trial_20261006T104025828291Z/plan.json) | Fixed schedule, settings, file fingerprints, and budget |
| [Training log](results/training_trial_20261006T104025828291Z/training_log.jsonl) | All 20 updates with IDs, labels, losses, timing, and memory |
| [Training result](results/training_trial_20261006T104025828291Z/training_result.json) | Actual step count, changed adapter fingerprints, and save information |
| [Final adapter](results/training_trial_20261006T104025828291Z/adapter/adapters.safetensors) | Separately saved learned additions |
| [Adapter configuration](results/training_trial_20261006T104025828291Z/adapter/adapter_config.json) | Structure needed to load those additions |
| [Reload check](results/training_trial_20261006T104025828291Z/reload_check.json) | Exact recovered weights and forward-output comparison |
| [Evaluation predictions](results/training_trial_20261006T104025828291Z/evaluation_predictions.jsonl) | Actual raw decisions on the fixed baseline sample |
| [Comparison](results/training_trial_20261006T104025828291Z/comparison.json) | Before/after metrics, worker outcomes, and final baseline check |

The update-10 checkpoint is retained in the same run's `checkpoint_0010` folder. It was not selected or evaluated using validation. Local run files remain excluded from ordinary Git commits.

## Run and understand the code

From the project folder, with access to the Mac GPU:

```bash
"model selection/.venv/bin/python" "model selection/train_trial.py"
```

This command performs a new bounded training trial and evaluation. It starts from the original model, saves into a new timestamped folder, and never overwrites this run. It is different from the earlier demo and dry probe: it **does apply real adapter updates**.

The five new offline tests use fictional records and no GPU:

```bash
"model selection/.venv/bin/python" -m unittest discover -s "model selection" -p test_train_trial.py -v
```

They passed checks for paired/reproducible scheduling, invalid update budgets and duplicate groups, changed preparation and failed probes, distinct timeout/failure outcomes, and interrupted prediction files. The actual local run separately tests gradients, updates, saving, reloading, and model inference.

The earlier [readiness report](TRAINING_READINESS.md) explains how the data and memory checks were completed without training, and the [baseline report](BASELINE.md) remains our fixed reference.
