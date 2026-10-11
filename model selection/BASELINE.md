# Our first development baseline

Recorded October 6, 2026. This measures **Qwen3-4B-Instruct-2507 before our fine-tuning**. A baseline is the starting result against which we compare later changes.

**Run status: completed.** All 120 planned decisions returned valid labels. Qwen matched **30 of 60 original-order labels (50.0%)**. This is now our first measured development baseline; no fine-tuning has run.

## The measured starting result

| Measure | Result |
| --- | ---: |
| Original-order preference agreement | 30/60 — 50.0% |
| Swapped-order agreement, translated back | 31/60 — 51.7% |
| Expected A, original order | 16/20 — 80.0% |
| Expected B, original order | 12/20 — 60.0% |
| Expected preference tie, original order | 2/20 — 10.0% |
| Same answer identity after swapping | 30/60 — 50.0% |
| Label matched in both orders | 19/60 — 31.7% |
| Invalid output / generation error / skipped input / missing result | 0 / 0 / 0 / 0 |

The model often forced a winner on examples labeled as ties: it predicted only four ties in the original order, and two of those matched a tie label. Changing answer positions changed the underlying selection on 30 questions. We have measured this instability; we have not established its cause.

In the original presentation Qwen chose displayed A 33 times, B 23 times, and tie four times. In the swapped presentation it chose displayed A 44 times, B nine times, and tie seven times. After mapping swapped choices back, those 44 displayed-A choices refer to original B. This distinction prevents accidentally reporting a change in letter as a change in preferred answer.

| Simple comparison on the same 60 original presentations | Matching labels |
| --- | ---: |
| Always choose A, always B, or always tie | 20/60 each — 33.3% |
| Choose the longer answer by character count | 26/60 — 43.3% |
| Choose the shorter answer by character count | 14/60 — 23.3% |
| Qwen with the fixed judging instructions | 30/60 — 50.0% |

Qwen matched four more labels than the longer-answer rule on this sample. That small difference does not establish a dependable advantage on new questions. A single-label rule gets exactly one third here because we deliberately balanced the labels.

| Local resource measurement | Observed value |
| --- | ---: |
| Complete worker run, including file verification and loading | 265.94 seconds, about 4 minutes 26 seconds |
| Model loading | 2.84 seconds |
| Median generation evaluation | 2.07 seconds |
| p95 generation evaluation | 3.21 seconds |
| Peak MLX allocation | 8.27 GiB |
| Actual input-token range | 254–2,191 |

All selected inputs fit; no example was truncated or replaced. These inference measurements do not demonstrate training memory fit. No paid API or Colab session was used.

[Actual summary](results/baseline_20261006T100253287762Z/summary.json) · [Raw predictions](results/baseline_20261006T100253287762Z/predictions.jsonl) · [Plan saved before predictions](results/baseline_20261006T100253287762Z/plan.json) · [Runtime measurements](results/baseline_20261006T100253287762Z/runtime.json)

## Three recorded disagreements to inspect later

These are differences from the publisher's preference labels, not independent findings that an answer is factually wrong. The IDs locate full examples in the saved sample. We did not alter the prompt or labels to make these cases pass.

| Example ID | Human preference | Original decision | Swapped decision, mapped back |
| --- | --- | --- | --- |
| `helpsteer2-preference-8707` | A | B | B |
| `helpsteer2-preference-8864` | B | A | B |
| `helpsteer2-preference-8854` | tie_preference | B | B |

## What happens next?

Keep this prompt, sample, and baseline results as a fixed comparison. Next measure the **training portion's** input lengths and verify a small training method's memory and compatibility. Those checks are still pending; this run measured only inference on the selected validation examples.

A later controlled training experiment should explicitly check ties and answer-order stability against this baseline. Do not connect this experimental judge to the product as a reliable evidence checker. The application continues using its existing quotation rules. The separate from-scratch learning path remains available.

## What are we measuring?

Can the existing Qwen model choose the same preferred answer as the public dataset's human label? Can it keep that choice when we exchange answer positions?

This is **preference agreement**, not a factual-truth score. HelpSteer2 does not provide separately verified evidence for these pairs. A preferred answer can still contain errors. Our product's quotation-checking rules solve a different, narrower task and are not being evaluated as a general preference judge here.

## What happens, in order?

1. Open only the prepared **validation** portion and check its saved file fingerprint. Calibration and final-test records are not opened.
2. Put examples in a repeatable order using their IDs. Choose one row per question group, stopping at 20 labels of A, 20 of B, and 20 preference ties.
3. Save the exact plan, sample, original judge instructions, code fingerprint, model revision, and dependency-list fingerprint before requesting predictions.
4. Verify the downloaded model files against their receipt, then load Qwen locally. No model files are downloaded and no paid API is called.
5. Send only the question, available evidence, and the two candidate answers. Keep the expected label and reviewer notes outside the model messages.
6. Run each question twice, once in the original order and once with A/B exchanged. Alternate which order runs first. That makes 120 planned decisions on **60 questions**, not 120 independent questions.
7. Accept only exactly `A`, `B`, or `tie_preference` after surrounding whitespace is removed. Extra explanations, missing results, errors, and capped outputs remain visible failures or unresolved results.
8. Translate swapped decisions back to the original answer identity, score agreement, and save every raw output plus timing and memory measurements.

For example, if original A was preferred, Qwen should say B when that same text is displayed second. After translation, both decisions refer to original A. If the translated decisions differ, the preference changed with answer order.

## The limits fixed before the run

| Setting | Value |
| --- | --- |
| Dataset | HelpSteer2 prepared v2; validation only, 396 rows available |
| Sample | 60 distinct question groups, balanced across the three labels |
| Selection | Sort by SHA-256 of `baseline-v1:` plus record ID, then take the first 20 per label with unique groups |
| Candidate input | Question, evidence field, original answer texts; no expected labels or reviewer explanations |
| Prompt | Existing `judge_instructions.txt`, copied into the run plan without changing it for this sample |
| Input ceiling | 4,096 tokens per presentation; a token is a small piece of text |
| Output ceiling | 16 tokens per presentation |
| Over-limit input | Mark `input_too_long`, do not truncate, and do not replace the selected example |
| Generation | Temperature zero: select the highest-scoring next token rather than sample randomly |
| Worker deadline | 900 seconds, including imports, model-file checks, loading, and inference |
| Automatic retries | Zero |
| Spending | No paid model API calls; computation uses the local Mac |
| Training | None; model weights remain unchanged |

The supervising process stops the worker if its deadline expires and writes a report from the completed records. Any remaining planned decisions are `not_run`. A failed worker is not a completed zero-quality baseline.

## How to read the results

- **Original-order agreement:** matching labels divided by all 60 planned original questions. Unresolved outcomes stay in the denominator.
- **Swapped-order agreement:** the same measure after translating the decisions back to original answer identities.
- **Per-label agreement:** separate results for the 20 A, 20 B, and 20 tie examples.
- **Swap consistency:** how often both presentations select the same underlying answer, whether or not that agrees with the label. Missing or invalid pairs are reported separately.
- **Both orders match:** a stricter count requiring agreement with the label in both presentations.
- **Simple rules:** always A, always B, always tie, prefer the longer answer, and prefer the shorter answer. Length here means characters; equal lengths produce a tie. These are comparison points, not recommended judges.
- **Median time:** the middle generation time. **p95 time:** the time at or above 95% of this run's measured generation times. Both include prompt construction; neither includes initial model loading.
- **MLX peak memory:** peak allocation reported by the model's local calculation library. It is not total computer memory, and it does not establish that training will fit.

## Limits of the experiment

This is a small, deliberately balanced development sample. Its overall percentage gives equal weight to A, B, and ties; it does not estimate the natural label mix in the full dataset or real user traffic. It is not a confidence percentage for an individual answer.

Each label has only 20 questions: a single changed decision moves that category's score by five percentage points. We have one run, not an estimate of variation across repeated runs, prompts, or new samples. The paired orientations are dependent observations and must not be counted as 120 independent questions.

The labels are upstream human preferences, not new independent reviews by this project. Public-data exposure during Qwen's earlier pretraining is unknown. The earlier overlap audit cannot exclude every paraphrase or shared source. No factual-grounding, safety, live-provider, synthesis, or end-to-end product-quality result is established here.

We keep the prompt fixed for this run and retain mistakes. Later tuning may use validation, but those tuned results must still be called development results. The final test remains reserved for a fixed version.

## Reproduce the run

From the project folder, using a terminal with access to the Mac GPU:

```bash
"model selection/.venv/bin/python" "model selection/baseline.py"
```

This creates a new timestamped folder under `model selection/results/`. It selects the same sample while the dataset and selection code are unchanged. The model library needs Metal GPU access; the first sandboxed attempt failed before making any predictions. That [failed summary](results/baseline_20261006T100241849311Z/summary.json) is retained separately.

A run folder contains:

| File | Meaning |
| --- | --- |
| `plan.json` | Model version, fixed prompt, dataset/code fingerprints, limits, and experiment rules |
| `sample.jsonl` | The selected public examples and labels for scoring; labels are excluded from prompts |
| `predictions.jsonl` | Raw model output, original-identity decision, stopping reason, status, token counts, elapsed seconds, and memory |
| `runtime.json` | Verified model-file receipt status, loading time, and MLX memory |
| `summary.json` | Worker outcome, aggregate scores, simple comparisons, and unresolved outcomes |

The files remain local and the results folder is already excluded from Git. Original datasets and earlier teaching results are preserved.

## Check the measuring code without running the model

```bash
"model selection/.venv/bin/python" -m unittest discover -s "model selection" -p test_baseline.py -v
```

Seven tests passed. They check deterministic selection and unique groups, rejection of the wrong data portion, hidden answer keys, strict output parsing, swapped identities, failure/missing-result accounting, duplicate-result rejection, and Unicode-safe JSONL reading. They use fictional records and do not load Qwen. Passing these tests checks the measuring program; it does not certify the model's judgments.
