# Build status

**Current state, October 9:** version 1 runs all 13 steps live on this computer (see [README.md](../README.md)): web search plus saved documents, cited answers from up to three models, a model-based check of every sentence against all of its cited passages, the fine-tuned judge, a final recheck and a local web page. Six bugs found in an October 9 code review are fixed, each with a test that fails on the earlier code. The suite has 121 offline tests.

**Judge:** agrees with the human preference on 252/311 (81.0%) clear validation pairs (unchanged Qwen: 238/311, 76.5%). The full-data A100 run reached 243/311 (78.1%) and was not adopted. This is agreement with people's preferences, not the factual accuracy of the product, which has not been measured. The reserved final test has not been run.

**Previous milestone complete, October 7:** all 450 corrected Colab updates, six monitor checks, checkpoint selection and both full-validation evaluations passed. Selected `2e-5`, update 150: **252/311 clear matches (81.0%)**, versus **238/311 (76.5%)** for the matching CUDA/FP32 baseline; log-loss **0.450 versus 1.169**. The predefined probability-quality rule passes, justifying a longer 2,000-pair experiment, which has not started. The accuracy-test cutoff was not met (McNemar p = 0.054). [Measured results and limits](<../model selection/CORRECTED_JUDGE_COLAB_RESULTS.md>).

**Evidence saved:** exact scores, reports and source/data fingerprints passed the local audit; checkpoints remain on Drive. The planned four-step run is complete. Calibration fitting, final-test model evaluation and product integration have not run. Historical baseline v2 remains locked at 239/311.

**2026-10-06 roadmap update:** We replaced the 29-tie review with measurement-first development. Baseline v2 now uses all 396 validation pairs, A/B probabilities and two-order averaging, with 311 clear labels as the main grade and 85 ties kept diagnostic. All four fixed-weight evaluations and their audit completed: 3,168 valid score reads, with no new training. Original Qwen JSON matched 239/311 clear preferences (76.8%); plain text matched 236/311 (75.9%). The preregistered rule selects **JSON** for the next binary-training stage. Read [the measured results](<../model selection/BASELINE_V2.md>), [the comparable results table](<../model selection/EXPERIMENTS_V2.md>) and [the revised roadmap](<../model selection/JUDGE_ROADMAP_V2.md>).

**Current data release:** v2 contains 8,563 public preference pairs after a model-assisted sample review and a defined text-overlap audit. No crossing remains under those text rules; unseen paraphrases may remain. The 20 fictional practice examples remain ineligible for training. **Full-length readiness and the learning-rate sweep passed their execution checks. The selected adapter passes the probability-quality development screen; a longer experiment is justified but not launched.** Separate 20-update and 200-update experimental adapters exist; neither is integrated into the product. The dated September 30 section below records the earlier v1 stage.

Updated: 2026-10-09. Package version 0.1.0; product release `v1` (October 8).

**Working end-to-end scope (v1):** question → saved and/or web evidence → concurrent cited answers → sentence checks against the cited passages → eligibility → trained judge → selection or abstention → final recheck → answer with sources, checks, trace and cost → optional feedback record. The original document-quotation demo still runs offline with saved answers.

## Implementation coverage

This table describes the live v1 path (`answerlab/pipeline_v1.py`). The original quotation demo (`answerlab/pipeline.py`) is unchanged.

| Step | Working and locally tested | Pending |
| --- | --- | --- |
| 1 | English-first rule task labels, reasons, basic clarification | Learned classifier, reliable multilingual/risk/ambiguity handling |
| 2 | Source/freshness rules: saved documents, fresh web, or both when unclear | Learned decision, cache age policies |
| 3 | Answer models in configured priority order (the first `max_calls` with a key), verifier choice, per-question spending cap | Empirical provider choice, learned router |
| 4 | Concurrent cited-answer calls with timeouts, independent failures and a cost preflight against the budget | Paid integration tests, retries |
| 5 | One bounded, priced Tavily web search (sent only if the budget also covers the largest answer and verifier calls that can follow) plus keyword search over saved documents, trimmed to fit the prompt | Source authority policies, larger index |
| 6 | Pydantic records and OpenAI/Anthropic response mappings | Gemini/DeepSeek/Perplexity adapters |
| 7 | Sentences with their citations | General atomic-claim extraction |
| 8 | One batched checker call per answer labels each sentence against all of its cited passages; a word-for-word copy skips the call only when every cited passage matches | Measured checker error rate, source reliability, math/code checks |
| 9 | Eligibility (no contradicted sentence, at least 80% supported), then the local fine-tuned judge in both answer orders; exact ties are kept and broken by a named rule; rule-based fallback when the judge is unavailable | Evidence-grounded judge, calibration |
| 10 | Original-answer selection or abstention | Synthesis and bounded revision |
| 11 | The outgoing text and the fingerprint of every cited passage must match what was checked | Rechecking synthesized text |
| 12 | Local web page and CLI: answer, sources, per-sentence checks, judge readings, trace, timing and cost (including web search) | Public hosting, calibrated probabilities |
| 13 | Opt-in feedback with consent flags and pending review | Review/export tools, training, evaluated releases/rollback |

## Local judge baseline added October 6

The standalone [baseline report](<../model selection/BASELINE.md>) records a fixed 60-question validation sample and 120 actual local Qwen decisions. It is separate from the product pipeline. Seven evaluation-bookkeeping tests pass. The baseline is now locked and the separate dry training-readiness probe passed. Separate 20-update and 200-update adapters passed saving, reload, and development comparison. Broader validation, evidence-grounding evaluation, and product integration remain pending.

## First real LoRA trial added October 6

The [training-trial report](<../model selection/TRAINING_TRIAL.md>) records 20 actual adapter updates and 120 new validation decisions. Five new offline tests pass. The base model and baseline remain locked and unchanged. The preference gain comes with a clear-winner regression; this adapter remains experimental.

## Expanded LoRA trial added October 6

The [expanded-trial report](<../model selection/EXPANDED_TRIAL.md>) records 200 updates across 100 training groups and another 120 validation decisions. Its update-20 adapter exactly reproduces the earlier experiment. Original-order matches: 34/60; ties: 6/20; clear A/B winners: 28/40; answer-order consistency: 35/60. Four new offline tests passed, and an independent saved-artifact audit passed. These are development preference measurements, not final-test or factual-truth results.

## Fixed-weight prompt experiment added October 6

The [prompt comparison](<../model selection/TIE_PROMPT_EXPERIMENT.md>) made 240 valid new decisions across the original Qwen and saved 200-update adapter. Execution, six new offline tests, and independent artifact checks passed; the candidate failed the preselected quality screen. No model was trained or integrated.

## Validation performed for v1 and the October 9 fixes

- 121 automated tests pass with no network access (simulated providers, search and judge).
- Each of the six review findings was first reproduced on the unchanged code; the new tests fail on that code and pass on the fixed code.
- An independent review of the fixes found five further gaps (search costs after failures or cancellation, a search the budget could not follow up, an oversized number in a judge reply, small NVIDIA GPUs running out of memory); all are fixed and tested.
- Not yet measured: the checker's error rate, source reliability, answer completeness and the product's factual accuracy.

## Validation performed at the prototype milestone

- 29 automated tests passed using the existing virtual environment.
- Real MCP subprocess retrieval matched direct retrieval against the saved corpus.
- HTTP integration tests used simulated OpenAI/Anthropic replies; two mocked requests overlap.
- The fictional Device Z demo selects the unchanged quotation and reports zero fixture cost.
- At the prototype milestone, no real provider calls, paid search, dataset downloads, training, deployment, or purchases occurred. Public dataset preparation began later, as described below.

## Dataset preparation added September 30

HelpSteer2-Preference is downloaded and normalized: 9,121 examples after four exclusions, separated into train/validation/calibration/test by normalized question group while respecting official validation. Nine preparation tests pass. A first-100 training-side review view is available in [Training Data](<../Training Data/README.md>). Twenty fictional practice examples were also drafted and need human review; semantic duplicate/source overlap auditing and production training readiness remain pending. No model had been trained at that September 30 milestone.

The fictional sample was authored to demonstrate program behavior. It is not an independent benchmark and does not show that multiple LLMs outperform one.

## Product boundaries

The quotation checker can confirm copying fidelity while still returning an irrelevant or unreliable source quotation. Relevance uses simple overlap and completeness is unchecked. We display those limits. Do not describe this version as a general fact checker or deploy it as the proposed finished product.

Live use must be explicit. Price entries and spending allowance default to unusable/zero values; a request cannot silently start billing from the demo. In v1 the web search is priced too (`search_usd_per_call`) and counts toward each question's budget; live web search refuses to run without a price.

A v1 answer can be returned as *partly verified*: up to one sentence in five may be unconfirmed (never contradicted). Those sentences are marked in the answer, not removed, so they remain unchecked statements. Estimated dollar controls are not provider-guaranteed billing caps. Unknown usage and errors remain visible.

Question/source/answer storage is opt-in through `--save`. Feedback storage is a separate command. Keys and local run/configuration files are excluded from Git; `.env` is preserved and not automatically loaded.

## Current implementation milestone

Completed [baseline v2](<../model selection/BASELINE_V2.md>): all validation pairs, fixed existing models, binary probability scoring, two-order averaging, grouped intervals, paired tests, and the predefined plain-text format comparison. Original Qwen JSON matched 239/311 clear preferences (76.8%); plain text matched 236/311 (75.9%). The preregistered rule selects **JSON** for the next binary-training stage. The former 29-tie review is dropped. The [revised roadmap](<../model selection/JUDGE_ROADMAP_V2.md>) places broader training after this measurement work; calibration, final test, and product integration remain later.
