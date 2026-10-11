# Step 1: prepare examples for our judge

**Learning-rate sweep completed, October 7:** all three rates and full validation are finished. The selected `2e-5` adapter at update 150 reached **246/311 (79.1%)**, with log-loss 0.454 versus baseline 1.160. The probability-quality development screen passes; the accuracy screen does not. A bounded 2,000-pair experiment is justified but not launched. Read [the result and limitations](<../model selection/LEARNING_RATE_SWEEP_RESULTS.md>). Calibration/test remain reserved.

**Current October 6 update:** follow the [revised judge roadmap](<../model selection/JUDGE_ROADMAP_V2.md>). Baseline v2 completed all 3,168 score reads on all validation pairs with fixed weights and passed its artifact audit. Original Qwen JSON matched 239/311 clear preferences (76.8%); plain text matched 236/311 (75.9%). The preregistered rule selects **JSON** for the next binary-training stage. Before the next training stage, read [the protocol](<../model selection/BASELINE_V2_PROTOCOL.md>) and [run status](<../model selection/BASELINE_V2.md>). The former 29-tie content-review proposal is dropped. Calibration and final test remain reserved.

Updated: 2026-10-06. **The initial data audit (model-assisted sample review and text-overlap checks) is complete for an experimental preference release. Separate [20-update](<../model selection/TRAINING_TRIAL.md>) and [200-update](<../model selection/EXPANDED_TRIAL.md>) training comparisons have completed; this folder retains the source data and preparation history.** The model is now selected, its baseline is protected, and the [training-readiness checks](<../model selection/TRAINING_READINESS.md>) passed. The expanded update/save/reload comparison and [label-aware model-based error review](<../model selection/ERROR_REVIEW.md>) are complete. Validation review notes are not training data. The [240-decision prompt experiment](<../model selection/TIE_PROMPT_EXPERIMENT.md>) has completed with labels and weights fixed; its quality screen failed. No new training data or label changes were made. Calibration and final-test portions remain reserved.

## What the judge learns

We want a judge that reads a question, the information available, and candidate answers, then assesses them. A **dataset** is its collection of examples. A **label** is the assessment attached to an example.

For example: a fictional library opens at 9 am. Answer A says 9 am; answer B says 10 am. Our suggested label is A. The reason is the library's supplied opening time, not the model's name or the length of its answer.

This is preparation for a model that judges answers. Training a language model from zero is a separate project, and this collection would not replace the large text collection that it would need.

## What is already here?

| Collection | Actual status | Purpose |
| --- | --- | --- |
| HelpSteer2-Preference | Official pinned download: 9,125 human-labeled pairs | Learn general human answer preferences |
| Prepared HelpSteer2 v2 | 8,563 pairs after the initial four exclusions and 558 additional quarantines | General preference experiments; source-grounding is not labeled |
| First 100 public examples | A view of 100 training rows | Inspect format and labels; these are not 100 additional examples |
| Fictional practice examples | 20 model-written pairs, kept outside this repository | Rubric practice with sources; all need human review |

HelpSteer2 does **not** supply a separate, trusted evidence package for every pair. We store an empty evidence list and `not_annotated` for evidence checking. Its human preference labels are not proof that a favored answer is factually correct. We keep the two collections separate.

## Where to look

- [How to review an example](REVIEW_GUIDE.md): our plain-language marking rules.
- [Where public data comes from](SOURCES.md): direct links, dated choices, and reuse notes.
- [What each field means](FORMAT.md): the saved record explained.
- [First 100 public training rows](prepared/helpsteer2_v2/review_first_100.jsonl): real source data.
- [Preparation report](prepared/helpsteer2_v2/report.json): counts, exclusions, checksums, and limits.

JSONL means one complete saved example per line, which is convenient for programs.

## Why four separate portions?

| Portion | Count | Simple purpose |
| --- | ---: | --- |
| Training | 6,623 | Examples the model may practice on |
| Validation | 396 | Choose settings without using the final exam |
| Calibration | 775 | Later check whether confidence scores match observed reliability |
| Test | 769 | A held-out exam for a fixed experiment |

We preserved the publisher's validation allocation except for one overlapping question group. We assigned remaining publisher-training question groups approximately 80% to training, 10% to calibration, and 10% to test using a fixed text fingerprint. The exact counts differ from the percentages. A fingerprint is a repeatable identifier calculated from text; it does not understand meaning.

Three records belonged to one question group present in both official portions, so all three were excluded. One additional record had identical answers, so it was excluded from this first comparison collection. Exclusion records retain original line numbers and reasons; the raw download stays intact.

**Preparation is not a quality certificate.** The v2 audit additionally quarantined 551 rows connected across portions by the defined text checks and seven rows needing label follow-up. No crossings remain under those checks. Differently worded paraphrases and unidentified source families may remain. These public held-out examples are not our future fresh product benchmark. Do not use test answers to tune a judge. The review file draws only from training.

## Review status

The 20 fictional practice examples carry model-suggested labels only; they are not human-verified, and none is eligible for training.

The model-assisted sample review and defined overlap audit are complete. Qwen has been selected, evaluated, and used in 20-update and 200-update trials; the latter used all 100 prepared training groups in both orders. A larger permitted evidence collection and independent human review remain necessary for the eventual source-checking judge.

## The small preparation program

[prepare_helpsteer.py](prepare_helpsteer.py) reads local downloaded data, checks its saved fingerprint, converts the format, separates question groups, and writes the report. It makes no network requests and does not train a model.

This script reproduces the original v1 preparation, not the later v2 audit. The v2 report records its parent release and excluded IDs; reconstruct it by filtering those IDs from the unchanged v1 portions. Both versions are retained. To reproduce the original preparation into a new folder, run from the project root:

```bash
.venv/bin/python "Training Data/prepare_helpsteer.py" --output "Training Data/prepared/helpsteer2_recheck"
```

The program refuses to overwrite an existing output folder. Nine offline tests check label direction, missing evidence, duplicate handling, and question grouping:

```bash
.venv/bin/python -m unittest discover -s "Training Data" -p "test_*.py" -v
```

Raw and prepared public data are ignored by Git to avoid accidentally publishing downloaded material. The source card and download manifest are retained locally. Original application code, saved-demo examples, diagrams, and research PDFs are unchanged.
