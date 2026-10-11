# How we will know whether the product works

**Learning-rate sweep completed, October 7:** all 450 training updates, six monitor readings, and 396-pair full validation completed. The selected `2e-5` adapter at update 150 matched **246/311 clear preferences (79.1%)**, versus baseline v2 **239/311 (76.8%)**; log-loss improved from 1.160 to 0.454. The predefined probability-quality screen passes; the accuracy screen does not. A bounded 2,000-pair experiment is justified but has not started. Read [the measured result and limitations](<../model selection/LEARNING_RATE_SWEEP_RESULTS.md>). Calibration, final test, and product integration remain pending.

Status: the broader model/product evaluation protocol remains proposed. A [first local Qwen development baseline](<../model selection/BASELINE.md>) is now measured on 60 balanced validation questions, with simple-rule comparisons and swapped-answer checks. This is preference agreement, not a final test or end-to-end product-quality comparison. A [first 20-update LoRA trial](<../model selection/TRAINING_TRIAL.md>) now has a paired comparison on the same development sample; its aggregate agreement improved while clear A/B winner agreement declined. The [expanded 200-update trial](<../model selection/EXPANDED_TRIAL.md>) adds a third version: 34/60 original, 35/60 swapped, 6/20 ties, 28/40 clear winners, and 35/60 order-consistent choices. These repeated development checks do not replace a held-out final test. Broader testing remains pending. Numeric product release targets remain unset. The subsequent [error review](<../model selection/ERROR_REVIEW.md>) covers all 37 questions with a mismatch in either order and retains all original labels. The [one-sentence fixed-weight comparison](<../model selection/TIE_PROMPT_EXPERIMENT.md>) is now complete with 240 valid decisions. The 200-update adapter matched 35/60 original-order labels and 34/60 swapped-order labels, with 6/20 and 6/20 tie matches. Its preselected development screen failed. Saved old-prompt controls were reused; no repeated-run variation or final-test result is claimed. Model-based diagnostic notes and aggregate preference ties are not human-verified truth labels.

## Current measurement protocol

The selected [baseline-v2 protocol](<../model selection/BASELINE_V2_PROTOCOL.md>) replaces the small balanced sample as the main development measure. It uses all 396 pairs, 311 A/B labels for primary accuracy, conditional A/B probabilities, both answer orders, log-loss/AUC, grouped bootstrap intervals, paired tests, and separate tie diagnostics. V1 three-class scores remain historical and are not directly comparable. Read [the current v2 report](<../model selection/BASELINE_V2.md>) for actual execution status. Calibration and final test remain reserved.

## Ask two separate questions

1. Does our judge recognize good answers when we already have the candidates and evidence?
2. Does the complete product actually give users better answers for an acceptable cost and delay?

A judge can improve while the overall product gets worse because retrieval fails or synthesis introduces new mistakes. We must test both levels.

## Establish fair comparisons

| Comparison | What it tells us |
| --- | --- |
| One chosen model, ordinary prompting | The simplest product baseline |
| The same model with our retrieval | How much improvement comes from evidence alone |
| Two models plus a simple judge | Whether comparison adds value beyond retrieval |
| Two models plus task checks and an evidence-aware judge | Whether the proposed verification layer helps |
| Same setup with our trained judge | Whether training earns a place in the product |
| Best original answer versus synthesized answer | Whether rewriting adds useful information or introduces errors |
| Two-model parallel calls versus cheaper-first escalation | The cost and waiting-time trade-off |

Select the strongest single-model baseline using development data, then lock it before final testing. Also compare repeated samples from that same strong model with the multi-provider setup under the same total spending allowance. Otherwise we may mistake “spent more money” for “a better architecture.”

The system may never see the ideal winning label when answering. For analysis, a human can identify the best candidate afterward; that gives an upper bound on what selection alone could achieve. A perfect selector still cannot fix a question for which every candidate is wrong.

## What we measure

| Measure | Plain-language question |
| --- | --- |
| Answer correctness and usefulness | Did the answer solve the actual user request? |
| Claim support | What fraction of checked factual claims is supported by the right evidence? Report importance and unchecked claims too. |
| Citation precision | Of the cited claim-to-passage links, how many really support the linked claim? |
| Citation coverage | How many important claims needing evidence have supporting citations? |
| Retrieval coverage | When relevant evidence exists in the test corpus, did retrieval find it? |
| Preference agreement | How often does the judge agree with independent reviewers, including ties and neither-adequate cases? |
| Objective outcomes | Did applicable arithmetic, unit checks, or hidden code tests pass? |
| Answer rate and errors among answered requests | Does quality look high only because the system refuses most questions? |
| Synthesis regressions | How often did the new answer introduce an error absent from the selected supported material? |
| Cost per request | All model, tool, judge, retry, retrieval, and synthesis costs, including failed attempts |
| Waiting time | Typical time and p95: the time within which roughly 95% of requests finish |
| Operational failures | Timeouts, provider failures, malformed results, retrieval failure, and exhausted budgets |

Keep completeness alongside factual support: an answer that says almost nothing can avoid false claims while failing the user. Link validity is a separate mechanical check from whether the linked text supports the claim.

## Make the test honest

Keep four purposes separate: training teaches weights; validation chooses prompts and settings; calibration maps scores to observed reliability; the final test measures a locked system. Split by question family, source document, and related versions before forming answer pairs. Remove cross-dataset duplicates and document known overlap.

Use the same evidence snapshot, candidate conditions, and budget rules for comparisons intended to isolate one component. For live-web evaluation, save allowed source snapshots/references and timestamps so we know what was available when the answer was produced. Add later questions to measure behavior when facts and providers change.

Reviewers should not see provider identities or the system's preferred answer before labeling. Use a clear rubric, retain disagreements, and involve someone with the relevant knowledge for specialized questions. Do not use the same automatic judge as both the component being tested and the sole authority declaring it improved.

Report sample counts, results by task type, repeated-run variation where relevant, and uncertainty ranges such as confidence intervals. Tiny improvements on a small test set are not reliable evidence of superiority. Public benchmark contamination may remain unknown; fresh product tests matter.

## Challenge cases we need

- All providers repeat the same wrong statement.
- A short correct answer competes with a long convincing but wrong answer.
- Both answers are valid; both answers are invalid; one is incomplete.
- A real link points to a page that never supports the claim.
- A cited source is stale, inaccessible, copied from another source, or conflicts with stronger evidence.
- A retrieved page or candidate answer tells the judge to change its rules or award a high score.
- Swapping A/B positions changes the judge's preference.
- Evidence contradicts familiar model knowledge or uses changed fictional facts.
- Arithmetic is right but the units or interpretation are wrong.
- Code passes the supplied examples but fails independent edge cases.
- A provider times out; a tool returns malformed data; every provider fails.
- Synthesis drops an important exception or merges incompatible dates.

## Scoring and uncertainty

Use relevant hard requirements first. If a central calculation fails or the evidence contradicts the central claim, stylistic points cannot compensate. When factual truth cannot be established, record unknown instead of converting it into a numerical pass. A short answer can omit a false claim and still need a completeness penalty.

Start with understandable dimension labels and a brief reason. Do not choose arbitrary universal weights and call the result a truth probability. Fit weights or thresholds using validation data only, and check their behavior by task type.

Initially report reasons such as “two relevant sources support the main claim, but the newer figure could not be checked.” If we later show percentages, define the event being predicted, calibrate on separate labeled data, and measure reliability on held-out questions. Judge confidence, agreement between models, and answer correctness are distinct quantities.

## Conditions for advancing

| Move | Evidence required |
| --- | --- |
| Saved examples → APIs | The small program is understandable; credentials, usage recording, and failure handling are designed |
| Prototype → invited pilot | Defined scope; baseline comparison; source tracing; budget/deadline enforcement; meaningful failure tests; appropriate data handling |
| Prompted judge → trained judge | Equal or better measured quality at an acceptable cost/speed trade-off, with no unacceptable failure-category regression |
| Selection → synthesis | Added usefulness exceeds measured new-error risk; outgoing claims are rechecked |
| Fixed routing → learned routing | Savings persist on held-out questions while meeting the set quality floor |
| Invited pilot → wider release | Product targets are filled in, evaluated independently, and satisfied; there is monitoring and a tested rollback path |

Before a wider release, fill in: target audience, supported tasks, quality floor, unacceptable error categories, spending ceiling per request/month, p95 wait-time target, retention policy, and minimum evidence for a claimed improvement. These are intentionally unresolved instead of invented performance guarantees.

## Experiment record template

Record the question being tested, change from the baseline, hypothesis, dataset and split revisions, model IDs, prompts, evidence snapshot, checker versions, settings, total cost, sample count, results with uncertainty, examples of failures, and the decision to keep or retire the change.
