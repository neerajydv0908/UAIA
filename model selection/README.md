# Our first judge: Qwen3-4B-Instruct-2507

Selected on 2026-10-05. **This folder records Phase 2, the first and expanded Phase 3 training trials, and their Phase 4 development comparisons.** The completed setup is recorded in `FIRST_RUN.md`, and the first measured development baseline is recorded in [BASELINE.md](BASELINE.md). The setup, teaching-run, and baseline programs do not train the model.

**Current milestone, October 7:** the four numerical/data fixes and a [Colab Pro package](<CORRECTED_JUDGE_COLAB.md>) are prepared. Gradient clipping, bias-corrected Adam, and the FP32 A/B head passed local checks; selection now uses 200 training-side pairs outside the unchanged 2,000-pair pool. The next execution is **Colab GPU probe → corrected three-rate sweep → full validation against unchanged Qwen with the same scorer**. CUDA execution and longer training have not started. The older 79.1% result remains historical; calibration, final test and integration remain pending.

## What we are doing today

We chose a model that already understands text. Its files are stored locally, its teaching trial is complete, and its first 60-question development baseline is measured. Separate trials have now applied 20 and 200 updates to small LoRA additions and compared them with the unchanged baseline.

Think of the model as a student, the examples as a workbook, and our program as the instructions telling the student what to do. We have measured the student before and after a small amount of extra practice. The training-trial report shows the improvements and regressions. A subsequent prompt-only experiment changed one tie sentence, made 240 valid decisions with fixed weights, and failed its preselected quality criteria.

The actual model files go in `models/Qwen3-4B-Instruct-2507` inside this folder. The `.venv` folder holds separate tools for this model so the existing demo keeps its current tools. Download size is about 8 GB plus smaller support files; the actual receipt records exact bytes.

## Read in this order

1. [Why this model](Qwen3-4B-Instruct-2507.md).
2. [The small exercises](teaching_cases.json): question, source, A, B, and a teaching key.
3. [First-run status](FIRST_RUN.md).
4. [The first measured baseline](BASELINE.md): the fixed before-training result.
5. [Baseline protection](BASELINE_LOCK.md): how the reference is kept intact.
6. [Training readiness](TRAINING_READINESS.md): the completed length and dry memory checks.
7. [First real training trial](TRAINING_TRIAL.md): 20 actual updates, separate saving, exact reload, and the baseline comparison.
8. [Expanded trial](EXPANDED_TRIAL.md): 200 updates across all 100 prepared groups, exact update-20 reproduction, and the three-version comparison.
9. [Error review](ERROR_REVIEW.md): what the 37 mismatching questions reveal and the bounded next prompt experiment. [Detailed notes](ERROR_REVIEW_CASES.md) retain every reviewed case.

10. [Fixed-weight prompt experiment](TIE_PROMPT_EXPERIMENT.md): the completed 240-decision comparison, its failed criteria, and the preserved references.

11. [Baseline v2](BASELINE_V2.md): full-validation probability scores, uncertainty, diagnostic slices, and the completed input-format experiment.
12. [Comparable v2 experiments](EXPERIMENTS_V2.md): one table for the original model, existing adapters, format candidate, and simple rules.

## Run the small example

From the LLM project folder, use this single command after setup:

```bash
"model selection/.venv/bin/python" "model selection/try_judge.py"
```

The first quoted part selects Python with our model tools. The second quoted part selects our program. Quotes keep the space in `model selection` inside the same folder name.

That program reads the local model, sends three short fictional exercises, and prints its actual decisions. It saves a new result file each time. It uses no paid API, runs no training, and needs no model download after setup. Expecting A in a teaching key does not force the model to choose A: the key is withheld from its input.

The first two exercises reverse A and B. The third uses two equivalent answers. These exercises check that the setup runs. Even three correct answers would not establish accuracy on real questions.

## Repeating the setup later

```bash
python3 -m venv "model selection/.venv"
"model selection/.venv/bin/python" -m pip install -r "model selection/requirements.lock.txt"
"model selection/.venv/bin/python" "model selection/download_model.py"
```

Line 1 creates an isolated Python tool folder. Line 2 installs the versions recorded during this setup; this lock is for our tested Mac environment, not a Colab environment. Line 3 downloads the fixed public model version and checks the weight files. It may download about 8 GB if no files exist. Normal daily use needs only the earlier `try_judge.py` command. Python 3.13 was selected for this Mac setup.

## What happens after this?

The expanded trial is complete: original preference agreement is 34/60, versus baseline 30/60 and the first trial 36/60. Ties match on 6/20 and clear A/B winners on 28/40. Keep both adapters experimental and the original baseline fixed. The [error review](ERROR_REVIEW.md) is complete. The [prompt comparison](TIE_PROMPT_EXPERIMENT.md) has now completed. The 200-update adapter matched 35/60 original-order labels and 34/60 swapped-order labels, with 6/20 and 6/20 tie matches. Its quality screen failed. The user has replaced that next step with the [measurement-first v2 roadmap](JUDGE_ROADMAP_V2.md); the fixed-weight comparisons are complete and the broader trainer passed its memory gate; the monitored sweep is complete and supports planning a longer controlled experiment. Calibration and final-test portions remain reserved.

Small-scale updating, saving, and reloading now work. A useful general evidence judge and product integration remain later work. The setup, demo, baseline, and dry probe do not train; the `train_trial.py` and `expand_trial.py` commands do.
