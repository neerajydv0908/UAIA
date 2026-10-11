# Testing a clearer tie instruction with fixed models

October 6, 2026. **Completed: all 240 new decisions are valid. The candidate did not pass the development acceptance screen.** Execution and artifact checks passed; the quality requirements are a separate result.

We changed only **“equal” → “about the same”** in the tie instruction. A **prompt** is the instruction text given to the model. Model **weights** are its stored internal numbers; they stayed fixed. This experiment asked whether clearer wording alone helps, without further practice or training.

## What happened after what?

1. **Check the saved references.** Verify the protected original model/baseline, both earlier training trials, and the error review. Keep their files intact.
2. **Make one candidate instruction.** The old sentence was `Use tie_preference when their overall answer quality is equal.` The new sentence was `Use tie_preference when their overall answer quality is about the same.` Every other instruction stayed identical.
3. **Load the original Qwen.** Ask the same 60 validation questions in both answer orders: 120 new decisions.
4. **Load the saved 200-update adapter.** An adapter is a small set of learned additions attached to the original model. Load its saved numbers exactly, then ask those same 60 questions in both orders: another 120 decisions. Do not train either version.
5. **Translate swapped choices.** If original B moves into displayed A and the model answers A, score that as a choice of original B. A tie stays a tie.
6. **Compare with saved old-prompt decisions.** Match by question ID and order. Count improvements, regressions, tie recognition, clear-winner recognition, and order consistency.
7. **Apply the criteria chosen before the run.** Keep failed criteria visible; do not lower the requirements after seeing scores.
8. **Verify and save.** Recheck raw outputs, input fingerprints, unchanged labels and weights, earlier artifacts, and the arithmetic. Save the new run separately and record the decision below.

A **validation** set helps choose settings during development. These are repeated measurements on the same small set, not new unseen test questions. A label match means agreement with the dataset's human preference label; it does not certify factual truth.

## The measured comparison

The same 60 groups contain 20 A labels, 20 B labels, and 20 preference ties. “Clear” here means the saved label favors A or B; it is not a claim that every such preference is objectively certain. All swapped results below are mapped back to the original answer identity.

| Measure | Original Qwen: old | Original Qwen: new | 200-update adapter: old | 200-update adapter: new |
| --- | ---: | ---: | ---: | ---: |
| All label matches: original order /60 | 30 | 31 | 34 | 35 |
| All label matches: swapped order /60 | 31 | 32 | 35 | 34 |
| Tie matches: original order /20 | 2 | 3 | 6 | 6 |
| Tie matches: swapped order /20 | 5 | 5 | 8 | 6 |
| Clear A/B matches: original order /40 | 28 | 28 | 28 | 29 |
| Clear A/B matches: swapped order /40 | 26 | 27 | 27 | 28 |
| Same answer identity in both orders /60 | 30 | 34 | 35 | 35 |
| Matches label in both orders /60 | 19 | 21 | 23 | 25 |

The saved 20-update adapter was preserved but not evaluated again: the selected experiment concerned only the original Qwen and the saved 200-update adapter.

## Did it meet the preselected requirements?

All **240/240** planned decisions were completed in the allowed format; there were zero missing, invalid, over-limit, or failed decisions. The remaining requirements apply to the 200-update adapter.

| Requirement | Minimum selected before the run | Measured | Outcome |
| --- | ---: | ---: | --- |
| Original-order tie matches /20 | 7 | 6 | **Fail** |
| Swapped-order tie matches /20 | 9 | 6 | **Fail** |
| Original-order clear matches /40 | 28 | 29 | Pass |
| Swapped-order clear matches /40 | 27 | 28 | Pass |
| Same identity across orders /60 | 35 | 35 | Pass |
| Both orders match the label /60 | 23 | 25 | Pass |
| Original-order total matches /60 | 34 | 35 | Pass |
| Swapped-order total matches /60 | 35 | 34 | **Fail** |

**Failed requirements:** Original-order tie matches /20, Swapped-order tie matches /20, Swapped-order total matches /60. The screen is an exploratory development rule, not a statistical-significance test or product release qualification.

## Which existing decisions changed?

An **improvement** changes a previous mismatch into a label match. A **regression** changes a previous match into a mismatch. A decision can also change while remaining a mismatch, which is included under “both mismatch.”

| Model | Order | Improved | Regressed | Both match | Both mismatch | Unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original Qwen | Original | 1 | 0 | 30 | 29 | 0 |
| Original Qwen | Swapped | 1 | 0 | 31 | 28 | 0 |
| 200-update adapter | Original | 2 | 1 | 33 | 24 | 0 |
| 200-update adapter | Swapped | 1 | 2 | 33 | 24 | 0 |

The [changed-decision list](results/tie_prompt_20261006T230223795161Z/changed_decisions.json) records each affected question, order, original label, old choice, new choice, and change type. All publisher labels remain unchanged. These validation records are not training examples.

### The three adapter regressions

These are all decisions that previously matched the label and now do not. Each question and both candidates were read to identify the topic; the table reports preference changes, without treating the labels as verified facts.

| Case ID ending | Topic | Affected order | Old → new, mapped to original identities | Saved label |
| --- | --- | --- | --- | --- |
| 8747 | Website spelling tools given a URL | Original | A → B | A |
| 8710 | Types of life insurance | Swapped | tie → B | tie |
| 8928 | Comparing differences between strings | Swapped | tie → B | tie |

For 8710 and 8928, original order still chooses A while swapped order now chooses original B: both become “choose the first displayed answer” pairs. For 8747, both orders now choose original B, making the decision consistently wrong against the saved label. This illustrates why consistency and label agreement must be measured separately. The candidates contain claims without supplied evidence; no new external factual verification or relabeling was performed here.

## What happened to answer-position sensitivity?

| Behavior across both orders | Original Qwen: old | Original Qwen: new | 200-update adapter: old | 200-update adapter: new |
| --- | ---: | ---: | ---: | ---: |
| Always chooses the first displayed answer | 21 | 20 | 12 | 14 |
| Always chooses the second displayed answer | 2 | 1 | 2 | 2 |
| Changes between a tie and a winner | 7 | 5 | 11 | 9 |
| Total order disagreements | 30 | 26 | 25 | 25 |

A first-position choice means the model selects displayed A both times, even though the underlying answer has changed. These counts describe observed behavior; they do not establish why the model made a choice.

## Decision and next step

**Do not adopt this wording as the tie fix, and do not rebuild training conversations with it on the strength of this experiment.** The adapter's original tie count stayed at 6/20 and its swapped tie count fell from 8/20 to 6/20. It actually produced fewer tie replies overall: 21 → 17 across 120 decisions. Clear-winner matches improved by one in each order, while total matches across both orders remained 69/120. That trade-off fails the chosen objective.

The original Qwen improved slightly: original matches 30 → 31, swapped matches 31 → 32, and consistency 30 → 34. That is useful diagnostic evidence, but it does not rescue the failed adapter criteria or establish a general improvement. Changing the sentence alone did not resolve the measured tie problem in this saved adapter; this does not prove that training with the new wording would fail.

**Next proposed step:** inspect the answer content in the 29 already-prepared training tie groups and record what distinguishes them from clear-winner groups. The previous review inspected their released reviewer votes; it did not provide a full content audit of these training pairs. Use that training-only audit to choose one justified data or training change, keeping validation notes out of the teaching examples. No learning-rate, class-weight, label, prompt-training, or update-count change is selected or run now.

The original instruction and both saved adapters remain available as experimental references. The product still uses its quotation-checking rules. This experiment does not pretrain a new foundation model.

## Limits and runtime

Only the selected sentence changed in the actual messages. The question, evidence field, answer text, labels, tokenizer, answer parser, and model versions stayed fixed. Each model used 4,096 input tokens maximum, 16 output tokens maximum, greedy decoding (temperature zero), alternating answer-order execution, a 900-second worker deadline, and no automatic retries. A token is a small text piece read or generated by the model. Inputs were not truncated. The new sentence added exactly two input tokens to every one of the 240 decisions.

| Worker | Total seconds | Median decision seconds | p95 decision seconds | Peak MLX GiB |
| --- | ---: | ---: | ---: | ---: |
| Original Qwen | 255.68 | 1.92 | 3.15 | 8.27 |
| 200-update adapter | 253.17 | 1.92 | 3.18 | 8.42 |

p95 is the recorded time within which about 95% of decisions finished. Worker time includes checks, loading, and inference. MLX allocation is not total computer memory. No downloads, paid API calls, training updates, or weight-file writes occurred. Calibration and final-test examples were not opened.

The controls were saved earlier, as specified in the [preselected plan](results/error_review_20261006T221309230086Z/next_experiment_plan.json); they were not regenerated. The run's copy retains that proposal's historical `proposed_not_run` status, while `comparison.json` records completed execution. We did not measure repeat-run variation. These 240 decisions contain repeated orders and model versions of only 60 questions; they are not 240 independent test questions. Results cannot establish general improvement or the effect of training with this new wording.

## Checks and reproducible records

- Six new offline tests passed: exact-change enforcement, incorrect identities and duplicates, missing/invalid decisions, clear-winner losses despite extra ties, and historical-file/receipt changes.
- All 34 installed dependency versions match the protected lock file.
- All 35 protected baseline files, 14 first-trial artifacts, 52 expanded-trial artifacts, and six error-review artifacts matched their saved fingerprints. Earlier receipts also stayed unchanged.
- All 240 raw replies were independently reparsed and their labels, order mapping, unique identities, counts, and scores checked. Exact message fingerprints confirmed the intended prompt and unchanged question/answer payload. Both workers stayed within their deadlines.
- The saved 200-update adapter's loaded numerical fingerprint matched its training result before and after inference; the original model files remained protected and unchanged.

Read the [saved plan](results/tie_prompt_20261006T230223795161Z/plan.json), [comparison](results/tie_prompt_20261006T230223795161Z/comparison.json), [independent audit](results/tie_prompt_20261006T230223795161Z/independent_audit.json), and [verification receipt](results/tie_prompt_20261006T230223795161Z/prompt_verification.json). Raw results are under [original Qwen](results/tie_prompt_20261006T230223795161Z/baseline/predictions.jsonl) and [200-update adapter](results/tie_prompt_20261006T230223795161Z/updates200/predictions.jsonl).

To deliberately repeat this same 240-decision experiment in a new result folder, run from the project root:

```bash
"model selection/.venv/bin/python" "model selection/compare_tie_prompt.py"
```

That command performs real local inference and can use the GPU for up to fifteen minutes per model. It does not train. The completed run above is already saved; reading it needs no rerun.
