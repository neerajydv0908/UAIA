# Step 9: Judge answers

**Learning-rate sweep completed, October 7:** all three rates and full validation are finished. The selected `2e-5` adapter at update 150 reached **246/311 (79.1%)**, with log-loss 0.454 versus baseline 1.160. The probability-quality development screen passes; the accuracy screen does not. A bounded 2,000-pair experiment is justified but not launched. Read [the result and limitations](<../../../model selection/LEARNING_RATE_SWEEP_RESULTS.md>). Calibration/test remain reserved.

**Current October 6 update:** follow the [revised judge roadmap](<../../../model selection/JUDGE_ROADMAP_V2.md>). Baseline v2 completed all 3,168 score reads on all validation pairs with fixed weights and passed its artifact audit. Original Qwen JSON matched 239/311 clear preferences (76.8%); plain text matched 236/311 (75.9%). The preregistered rule selects **JSON** for the next binary-training stage. Before the next training stage, read [the protocol](<../../../model selection/BASELINE_V2_PROTOCOL.md>) and [run status](<../../../model selection/BASELINE_V2.md>). The former 29-tie content-review proposal is dropped. Calibration and final test remain reserved.

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/09_answer_judging/METHODS.md)

## My preferred method

Required quotation checks before ranking; simple relevance overlap with a visible tie outcome.

## Model choice

Selected on 2026-10-05: **Qwen3-4B-Instruct-2507** as the starting model for our custom judge. The original files are downloaded, and a separate local teaching trial works. See [model selection](<../../../model selection/README.md>). The [first measured development baseline](<../../../model selection/BASELINE.md>) is complete: 30/60 original-order preference labels matched, including only 2/20 ties; 30/60 choices changed after swapping answer positions. The application still uses the rule baseline. The baseline is locked and the dry training-readiness probe passed. The 20-update and expanded 200-update trials are complete and remain experimental; product integration is pending.

## Why prefer this now?

A transparent baseline gives us something a learned judge must improve. A pretrained judge is not automatically reliable just because it produces a numeric score.

**Why the alternatives wait:** A judge ensemble and adaptive rubrics add calls. Skywork and Prometheus remain comparison candidates with their documented limitations. No model has won a project benchmark.

## What is actually implemented?

Rule eligibility and ranking implemented. A standalone prompted Qwen preference baseline now exists. An experimental trained adapter now has a fixed-sample comparison. A dependable evidence judge, product integration, semantic relevance/completeness checks, and calibrated scores remain pending.

September 30 update: [judge-data preparation](<../../../Training Data/README.md>) has started with HelpSteer2 preference examples and separate source-based practice drafts. Missing evidence is explicit; preference labels are not treated as factual-support labels. No judge training had run at that September 30 milestone.

[Read the relevant code](../../../answerlab/checks.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

AdaRubric motivates requirements before averaging; Judge Anything motivates independently evaluating the judge. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Two inadequate answers cause abstention; two equally ranked adequate quotations preserve a tie. Fixture results are not model-quality evidence.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

## First trained adapter — October 6

The [20-update trial](<../../../model selection/TRAINING_TRIAL.md>) saved and reloaded correctly. Preference agreement on the fixed original-order development sample rose from 30/60 to 36/60, but clear-winner matches fell from 28/40 to 24/40. Ties improved from 2/20 to 12/20. Keep the adapter experimental; the product continues using its quotation rules. The subsequent expanded trial is recorded below. The baseline remains protected and unchanged.

## Expanded adapter — October 6

The [200-update comparison](<../../../model selection/EXPANDED_TRIAL.md>) used all 100 prepared training groups in both orders. Its first 20 adapter updates reproduced the earlier checkpoint exactly; the final adapter saved and reloaded correctly. Original preference agreement is 34/60, ties 6/20, clear winners 28/40, and order consistency 35/60. The [saved-error review](<../../../model selection/ERROR_REVIEW.md>) is complete. The [fixed-weight prompt experiment](<../../../model selection/TIE_PROMPT_EXPERIMENT.md>) subsequently completed all 240 valid decisions. The 200-update adapter matched 35/60 original-order labels and 34/60 swapped-order labels, with 6/20 and 6/20 tie matches. The candidate failed its preselected screen; no further training change has been selected. This remains a general-preference experiment outside the product; it does not establish evidence-checking quality.
