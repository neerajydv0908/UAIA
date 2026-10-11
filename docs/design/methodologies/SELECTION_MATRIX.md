# Proposed choices across all 13 steps

Prepared: 2026-09-28. **Decision status: proposed; no option has won a project experiment yet.**

Implementation update: [MY Pick](<../my_picks/README.md>) records the baselines chosen for the first working prototype and distinguishes them from the longer-term recommendations below. Consult [BUILD_STATUS.md](../../build_status.md) for tested coverage.

There is no single best model or method for every question. Our first choice should be understandable, affordable within the budget we set, and easy to compare against alternatives.

## A coherent starting configuration

| Step | My proposed starting method | Alternatives worth testing later | Why start here? |
| --- | --- | --- | --- |
| [1](01_understand_question/SELECTION.md) | Clear rules; one small structured classifier for unclear cases | Learned text classifier; fine-tuned local model | Easy to inspect wrong routing decisions |
| [2](02_freshness_decision/SELECTION.md) | Explicit freshness/source rules with an “uncertain” outcome | Learned freshness decision; source-specific cache policy | Avoids guessing that model memory is current |
| [3](03_model_selection/SELECTION.md) | A configurable fixed pair of providers | Task rules; cheap-first cascade; learned router | Establishes a fair baseline before routing research |
| [4](04_concurrent_requests/SELECTION.md) | Python asynchronous calls with limits and independent failure records | Sequential debugging; LiteLLM; LangGraph; durable jobs | Small enough to understand and test |
| [5](05_evidence_retrieval/SELECTION.md) | Small saved corpus and keyword retrieval; one web search service later | Dense retrieval; hybrid retrieval plus reranking; graph retrieval | Separates learning to use evidence from the difficulty of finding it |
| [6](06_response_normalization/SELECTION.md) | Provider adapters plus Pydantic validation | Dataclasses; native structured outputs; audited repair | Common records without rewriting the answers |
| [7](07_claim_extraction/SELECTION.md) | Manual examples, then one structured claim-extraction prompt | Sentence rules; two-pass extraction; trained extractor | Preserves qualifiers and source links for inspection |
| [8](08_verification/SELECTION.md) | Citation/source checks and appropriate deterministic checks | NLI support model; LLM support checker; isolated code tests | Uses checks matched to the task |
| [9](09_answer_judging/SELECTION.md) | Required checks plus a blinded, evidence-aware pairwise judge | Dimension scores; judge ensemble; open reward model; our own judge | Makes the selection decision explainable |
| [10](10_selection_and_synthesis/SELECTION.md) | Keep an adequate original answer | Extractive combination; grounded synthesis; bounded revision | Avoids adding new claims during rewriting |
| [11](11_final_reverification/SELECTION.md) | Recheck the exact outgoing answer; reuse only valid unchanged checks | Changed-claim checks plus regression checks; independent critic | Makes the final text the unit being checked |
| [12](12_response_and_uncertainty/SELECTION.md) | Fixed answer template, linked evidence, plain uncertainty reasons | Per-claim source view; calibrated probabilities later | Useful without pretending we know exact confidence |
| [13](13_feedback_and_learning/SELECTION.md) | Optional feedback, human review, offline evaluation | Small learned scorer; fine-tuned text judge; scratch-model learning; learned routing | Improves deliberately rather than learning every user vote |

A practical first provider pair to compare is OpenAI's general-purpose tier plus Claude's middle tier, with Gemini as a replacement/comparison. This is a proposed experiment, not a claim that this pair is best. Exact documented candidates and other providers are in the [catalog](MODEL_CATALOG.md). Budget, account access, actual outcomes, and acceptable waiting time decide the final pair.

## What the papers contribute

| Paper | Steps most directly influenced | Our proposed experiment | What the paper does not establish |
| --- | --- | --- | --- |
| AdaRubric | 1, 8, 9, 13 | Generic judging vs task templates vs adaptive rubrics | That our judge's confidence proves correctness |
| Optimal Model Activation Policies | 2, 3, 4, 9 | One model vs two in parallel vs cheap-first escalation | Optimal routing for five arbitrary providers |
| PDE-Agents | 5, 8, 11 | No evidence vs supplied relevant evidence vs retrieved evidence | That we need a graph database or that completion means correctness |
| Judge Anything | 7, 9, 10, 12, 13 | Measure the judge, compare rubric styles, test ties and failures | Universal superiority of a judge across tasks or modalities |

These are adaptations of the [reviewed papers](../../research_papers.md). Provider blinding, answer-order swaps, exact-output version checks, and our proposed feedback pipeline are project design controls; do not attribute them all to the papers.

## Build order after methodology selection

1. **Understand manually:** one question, one short source, two saved answers, one explained decision.
2. **Build a small local program:** load those records and show the decision. Learn Python and record formats.
3. **Add generation:** one API, then two concurrent APIs, preserving usage and failure status.
4. **Add evidence and checking:** saved passages, claim extraction, applicable checks, a prompted judge, selected answer, final check, and a simple output.
5. **Add fresh search and MCP:** one controlled search connection, then measure whether MCP makes integration clearer.
6. **Evaluate and pilot:** compare with a strong single model, publish limitations to testers, enforce budgets and data choices.
7. **Run separate improvement experiments:** synthesis, our judge, richer retrieval, and routing. Keep only demonstrated improvements.

The 13 responsibilities do not imply 13 separately deployed services. Initially, small Python modules and shared records are sufficient.

## How we choose a winner

Use the same question set, evidence snapshot, candidate conditions, and total spending allowance for each relevant comparison. Measure final correctness, completeness, source support, abstention rate, cost, and waiting time. Also test the individual component: for example, whether retrieval found the right passage.

Set acceptable quality losses and spending limits before looking at final test results. Choose settings using development data; reserve a separate calibration set if mapping scores to probabilities, and a final untouched test set. See [EVALUATION.md](../../evaluation.md).

All call types count: classifiers, generators, claim extraction, support checking, judges, search, retries, and synthesis. A “two-model system” may make more than two model calls in total.

## Decision record to fill in for each step

- Method and version:
- Why it fits the first users:
- Alternative being compared:
- Dataset/split and evidence snapshot:
- Quality, cost, and time limits set before testing:
- Result, including failure cases:
- Decision: keep / revise / defer:
- Reason and next review trigger:

We have not invented numeric launch targets, monthly bills, or hardware requirements.
