# Technical gist: how this project will work

Updated for the completed Colab sweep, full validation and longer-training decision on October 7, 2026.

## 1. What are we building?

We are building an app that **asks multiple AI models a question, checks their answers against evidence, and returns a well-supported answer**.

An **LLM (large language model)** is an AI model that reads and generates text. Different LLMs can give different answers to the same question. Our app will coordinate them and check their work.

Imagine two writers, a researcher, a checker, and an editor. Our Python program organizes those jobs. **Python** is the programming language used to write the app's instructions.

The goal is to improve answer quality within a time and money limit. Whether our approach actually improves on one good model must be measured.

## 2. First, understand the three separate parts

| Part | What it does |
| --- | --- |
| **The product** | Receives the question, calls models, finds sources, runs checks, and displays the result. We are building this software. |
| **The judge** | Compares candidate answers. A candidate is simply an answer being considered. We are experimenting with fine-tuning Qwen3-4B-Instruct-2507 for this job. |
| **From-scratch learning** | A separate learning path where we build a tiny model starting with random internal numbers. This remains an optional learning exercise. |

**Fine-tuning** means giving an already-trained model additional training for a specific job. **Pretraining from scratch** means teaching a new model from the beginning, starting with random **weights**—the internal numbers that training adjusts.

The first product judge follows the fine-tuning route. The current app still uses simple Python judging rules.

## 3. What happens after a user asks a question?

This is the intended flow of the larger product. Some parts already work in a small demo; the status table later shows which ones.

```text
User asks a question
        ↓
Understand the task and set limits
        ↓
Choose models and gather relevant evidence
        ↓
Ask the selected models for answers
        ↓
Organize the answers and identify important statements
        ↓
Check those statements against evidence or suitable tools
        ↓
Compare answers and select an adequate one
        ↓
Check the final outgoing answer again
        ↓
Show the answer, sources, and remaining uncertainty
        ↓
Optional feedback → review → possible future training
```

The steps below follow execution order. In the older 13-step plan, retrieval is called Step 5, but it happens before model requests when the models need sources to answer.

### Step 1 — Read the question and identify the task

The app checks what the user needs: an explanation, a document answer, current information, a calculation, or something else.

For example:

> According to the supplied manual, what is Device Z's operating temperature range?

This is a document question. If an essential detail is missing, the app should ask for clarification before continuing.

### Step 2 — Decide what information is needed

For a manual question, use the supplied manual. For current information, the intended product needs fresh sources. A creative-writing request needs different checks.

The current prototype can search saved documents. Live web retrieval is future work, so a request needing it must remain visibly unresolved.

### Step 3 — Choose models and set limits

A **router** is the part that decides which models to ask. We begin with a configured pair; later choices should depend on measured quality, cost, and speed.

Before calls begin, set limits on spending, waiting time, and call count. Any later search, judging, retry, or rewriting must fit within the overall budget.

### Step 4 — Find the relevant source passages

**Retrieval** means searching a collection and selecting useful pieces of text. The selected passages become the **evidence** used to write and check answers.

For our fictional Device Z example, the saved manual says:

> Device Z operates from 0 to 40 degrees Celsius in dry conditions.

Keep the passage's source label and version so checks can point back to the actual text. Retrieved text is material to inspect; instructions inside it must not override the app's rules.

### Step 5 — Ask the selected models

In the planned live flow, send the question, relevant evidence, and answer instructions to the selected models.

An **API** is a connection that lets our program request something from another service. Model providers supply answers through APIs. We can make requests **concurrently**, meaning their waiting times overlap.

Start with two candidates, A and B. If one request times out, record that failure separately; it is not a completed answer or a passed check.

### Step 6 — Put the replies into a common format

Different providers return different response formats. An **adapter** translates each provider's response into our app's common record.

That record keeps the answer text, model identity, completion status, timing, and available usage information. Formatting the record must preserve the original answer's meaning.

### Step 7 — Identify the statements that need checking

A **claim** is a statement that can be checked. An answer might contain claims about a temperature range, operating conditions, and warranty length. Each may need different evidence.

Keep numbers, units, conditions, and source references attached to their claims. The current prototype handles complete quoted passages; understanding arbitrary prose is future work.

### Step 8 — Run the appropriate checks

A **verifier** is a check suited to the task: source comparison for document claims, calculation checks for arithmetic, or isolated tests for generated code.

The intended evidence checks should distinguish supported claims, contradicted claims, insufficient evidence, and checks that do not apply. Missing or unchecked information must never become a pass.

**Today, our checker only establishes whether a full quotation matches its named saved passage.** A matching quotation does not prove that the source itself is true. A working citation link alone does not prove support either. Math and code verifiers remain future work.

### Step 9 — Judge the answers using the checks

The intended evidence-aware judge receives the question, candidates, evidence, and checking results. It compares correctness, relevance, completeness, and clarity. A serious factual error should not be cancelled out by attractive writing.

We plan to hide provider names during judging and test whether swapping A/B positions changes decisions unfairly.

Today this job uses Python rules. Qwen is a separate experiment. The earlier 20-update and 200-update trials learned A, B, or a preference tie. The new measurement protocol reads A/B probabilities and averages both answer orders; future training will teach only A/B labels and produce a too-close-to-call outcome from the probability margin. Teaching reliable evidence judgments requires additional reviewed source-based data.

### Step 10 — Select an answer, or explain why none is adequate

Initially, return an adequate original candidate. Allow ties and **abstention**, which means saying there is not enough support to return a checked answer. Agreement between two models is not proof of truth.

Later, we may add **synthesis**: combining supported parts into a new answer. This is a separate experiment because combining text can introduce mistakes. Any repair attempt must have a fixed limit.

### Step 11 — Recheck the actual outgoing answer

Check the exact text that will be shown to the user, together with its sources. This matters especially after synthesis: a newly written sentence was not automatically checked when the original candidates were checked.

If an essential check fails or remains unresolved, explain the limitation or request clarification.

### Step 12 — Display the result and optionally collect feedback

Show the answer, relevant sources, and important uncertainty in simple language. Avoid invented confidence percentages.

With permission, save feedback for later review. A thumbs-up does not immediately train the judge or change its weights.

## 4. What does the working demo do today?

The default demo reads **saved, authored answers**. It does not ask live AI models to generate them.

| Saved item | Text or result |
| --- | --- |
| Manual | Device Z operates from 0 to 40 degrees Celsius in dry conditions. |
| Answer A | Device Z operates from 0 to 60 degrees Celsius in dry conditions. |
| Answer B | Device Z operates from 0 to 40 degrees Celsius in dry conditions. |
| Check A | `unknown`: the quotation differs from the saved passage. The current checker does not independently prove it false. |
| Check B | `pass`: the full quotation matches its named source. |
| Final result | Select B, recheck it, and display it with its source label and checking limits. |

Run this from the project folder in a terminal:

```bash
.venv/bin/python -m answerlab demo
```

This uses the project's prepared Python environment and starts the saved-example demo. It needs no API key or paid call. The command was rerun while preparing this guide and selected B as described.

## 5. How will our judge learn?

Training is a separate development process. It does not happen each time someone asks a question.

1. **Prepare examples.** A dataset is a collection of examples. Each preference example contains a question, A, B, and a reviewed expected choice called a label. Evidence exercises also need sources and suitable labels.
2. **Measure the starting model.** A baseline is a result recorded before our change. Measure the existing Qwen model and simple rules before training.
3. **Separate practice from evaluation.** Keep related questions and shared sources together when dividing the data, so practice answers do not leak into evaluation.
4. **Fine-tune a small version.** Show the inputs, compare the model's prediction with the expected label, and adjust trainable weights. The label teaches the model; it must not appear in the question it is asked to judge.
5. **Compare before and after.** Check decisions, ties, failures, speed, and memory. Use separate final-test examples after settings are fixed.
6. **Integrate only if useful.** Connect the evaluated judge to the app if its measured results justify doing so. Later feedback can enter another reviewed training release.

The four data portions have different jobs:

| Portion | Purpose |
| --- | --- |
| Training | Teach the model. |
| Validation | Choose prompts and training settings. |
| Calibration | If we later report confidence, check how scores relate to observed reliability. |
| Final test | Evaluate a fixed version using examples kept out of development. |

Our current HelpSteer2 collection teaches **human answer preference**. Preference alone does not establish factual truth or source support. The fictional evidence exercises still need independent human review and remain ineligible for training.

### How do we now measure the judge, step by step?

1. Give it one saved validation question and two candidate answers. Keep the expected winner hidden from the model.
2. At the point where it would write its choice, read its **logits**: raw numerical scores for the next token. Convert the A and B scores into an A/B probability.
3. Swap the answers and repeat. Translate the second result back to the original answer names before combining it with the first result.
4. Average the two probabilities. For example, if the original order gives original A a score of 0.80, and the reversed order gives displayed A a score of 0.30, then original A received 0.70 in that reversed reading. Its average is `(0.80 + 0.70) / 2 = 0.75`. These are invented numbers to explain the calculation.
5. Compare the chosen answer with the saved label on all 311 clear-winner pairs. Separately inspect the 85 tie-labelled pairs, stronger preferences, shorter-answer winners, and disagreements between orders.
6. Report accuracy, **log-loss** (how strongly wrong choices are penalized), and **AUC** (how well the probabilities rank the two label groups). Use repeated resampling of whole question groups to show uncertainty intervals.
7. Use the rule written before the run to decide whether a change helped enough. The completed sweep passes the probability-quality rule, so planning the next controlled training experiment is justified.

A score of 0.75 here means relative preference for A among the two choices. It is **not a measured 75% chance that A is factually correct**. A margin near 0.50 can produce “too close to call”; final confidence calibration is a later step using its own reserved data.

During later fine-tuning, **LoRA** means learning small additions to the original model's frozen weights. These additions are also called an **adapter**. That is different from the provider adapter in Step 6, which is ordinary software translating an API response.

## 6. Where does MCP fit?

**MCP (Model Context Protocol)** is a standard way for an app to connect to tools. Our existing local MCP tool searches saved documents.

```text
Our app → provider API → candidate answer
Our app → MCP document-search tool → source passages
Our app → checking and judging → selection
```

MCP provides a tool connection. The checking and judging logic still has to be built and evaluated.

## 7. What is finished, and what happens next?

| Piece | Current state |
| --- | --- |
| Saved-answer demo | Works from the terminal: retrieval, quotation checks, selection, final recheck, and display. |
| OpenAI and Anthropic connections | Code exists and simulated responses were tested. Paid live integration tests remain pending. |
| Local document retrieval | Works directly and through MCP. |
| Judge dataset | Version 2 contains 8,563 public preference pairs after the documented audit. This is an experimental collection. |
| Qwen3-4B-Instruct-2507 | Downloaded, then evaluated on 60 validation question groups in both answer orders. Original-order preference agreement: 30/60; ties: 2/20. Choices changed on 30/60 questions after swapping. See [the baseline report](<../model selection/BASELINE.md>). |
| Training preparation | Baseline locked; all training lengths measured; 100 groups prepared; the 2,048-token LoRA gradient/memory probe passed at 9.42 GiB. No weight updates were applied. |
| Our fine-tuned judge | Separate LoRA adapters completed 20 and 200 updates and reloaded exactly. With the original instruction, the 200-update adapter has development preference agreement of 34/60; ties 6/20; clear winners 28/40. Both remain experimental and outside the product. |
| Judge error review | All 37 questions with a mismatch in either order reviewed; labels remain unchanged. |
| Tie-prompt experiment | Completed 240 valid decisions with fixed weights. The 200-update adapter matched 35/60 original-order labels and 34/60 swapped-order labels, with 6/20 and 6/20 tie matches. The one-sentence change failed its development criteria; no training occurred. |
| Binary-training readiness | Full-length memory and exact restart checks now pass at 10.216 GiB under the 18-GiB guideline. [Measured fix](<../model selection/TRAINING_MEMORY.md>). |
| Learning-rate comparison | Completed all 450 updates and full validation. Selected `2e-5`, update 150: **246/311 clear matches (79.1%)**, log-loss **0.454**, AUC **0.878**. The probability-quality rule passes; accuracy improvement remains uncertain. [Measured result](<../model selection/LEARNING_RATE_SWEEP_RESULTS.md>). |
| Baseline v2 | Completed all 396 validation pairs in both orders for four fixed model/format versions: 3,168 valid score reads. Original Qwen JSON matched 239/311 clear preferences (76.8%); plain text matched 236/311 (75.9%). The preregistered rule selects **JSON** for the next binary-training stage. No new training. [Full result](<../model selection/BASELINE_V2.md>). |
| Corrected Colab judge | Completed 450 updates, six separate monitor checks and full validation. **252/311 (81.0%)** versus **238/311 (76.5%)** for the matching unchanged baseline. The probability-quality screen passes; the longer 2,000-pair experiment is justified but not launched. [Measured result](<../model selection/CORRECTED_JUDGE_COLAB_RESULTS.md>). |
| Web search, general factual checking, synthesis, public interface | Future work. |

The baseline is protected. Full-validation probability measurement, input-format comparison, and two-order averaging are complete. The binary trainer learns from the A/B decision and keeps both answer orders in each update. Its original memory check failed. We fixed the order of calculations: **calculate gradients → finish that calculation → check their size → collect eight learning signals → update and save**. The full-length check now passes at 10.216 GiB, and restarting reproduces the next update exactly. Read [the simple memory-fix explanation](<../model selection/TRAINING_MEMORY.md>).

The completed historical experiment ran like this: **start identical adapters → train each with a different learning rate → save every 25 updates → measure the same validation pairs → select the lowest log-loss checkpoint → evaluate it on full validation**. A learning rate controls how far the learned values move in one update. The sweep is complete. The middle learning rate (`2e-5`) won the fixed monitor comparison, then matched 246/311 clear validation preferences versus the original model's 239/311. Its probability scores improved much more clearly than its accuracy. That passes our predefined probability-quality rule and justifies a larger controlled experiment; it does not yet qualify the judge for the product. Read [the simple experiment walkthrough](<../model selection/LEARNING_RATE_SWEEP.md>).

Now completed: **setup and GPU checks → three learning rates → separate monitor chooses rate 0.00002, update 150 → full validation against unchanged Qwen → decision to try more data**. The corrected judge matched **252/311 clear preferences (81.0%)**, versus **238/311 (76.5%)** for the matching unchanged model. Probability errors decreased enough to pass our predefined development rule. [Measured result](<../model selection/CORRECTED_JUDGE_COLAB_RESULTS.md>). The longer run has not started. Calibration fitting, the one-time final-test evaluation, and product integration come later.

We must also compare the complete app with one strong model using the same spending allowance. A better judge does not automatically mean a better product: retrieval, answer generation, and final rewriting can each introduce errors.

For the architecture, see [main.md](overview.md).

## 8. What changed before the next training run?

1. **Read the score more precisely.** At the answer position, the model has a final hidden vector: numbers representing what it read. Multiply that vector by the A and B output rows using float32 arithmetic, then turn the two scores into a probability. This avoids rounding the final comparison in BF16 first.
2. **Collect learning signals from both answer orders.** Four question pairs produce eight examples. Average their gradients—the suggested changes to the learned numbers.
3. **Limit unusually large changes.** Measure the combined gradient's size and clip it to 1.0 before Adam uses it. Record the original size so spikes remain visible.
4. **Correct Adam's early estimates.** Bias correction accounts for the optimizer's initially empty history. This changes the recipe, so we compare learning rates again.
5. **Choose progress using a separate monitor.** Two hundred question pairs are held out of training, including later larger runs. They choose the best checkpoint. Full validation measures the selected result afterward.
6. **Try this on Colab's GPU.** First run a small check, then the controlled sweep. All nine tests passed in Colab. The real-model GPU check also passed: about 9 GiB of memory and an exact restart. The three-rate sweep and both full-validation evaluations are complete.
7. **Change one learning idea at a time.** Compare hard answers, gently softened answers, and actual annotator vote shares on the same data. Later try more data, length weighting, and readable formatting. Their quality is not measured yet.
8. **Calibrate only the final model.** Calibration adjusts confidence using a separate dataset. Until then, a high score is a useful ranking signal, not a proven probability of correctness.

The original model files and completed results remain preserved. This work trains the judge's small LoRA additions; it does not teach a foundation model language from scratch or replace the app's current judging rules.

## Colab setup note

The first Colab tests stopped because an installed image-processing package could not work with the selected PyTorch installation. Our judge reads text, so it does not need that image/audio package pair. The repair removes those add-ons, restarts the session, then reruns the tests. Training starts only after the tests and GPU probe pass. [Exact steps](<../model selection/CORRECTED_JUDGE_COLAB.md>). This setup failure does not measure the judge's quality.

## Completed Colab run, step by step

1. **Make training reliable.** Fixed setup, passed nine tests, then checked GPU memory and exact restart of saved training state.
2. **Compare three learning rates.** Each version saw the same 600 answer pairs in both orders. Together they completed 450 updates.
3. **Check learning on separate questions.** A monitor is a small set of questions withheld from training. All six checks used 200 training-side pairs outside the larger training pool.
4. **Choose a saved version.** The monitor selected learning rate 0.00002 at update 150. A checkpoint is a saved version of the learned model additions.
5. **Measure against the starting model.** Both models answered all 396 validation pairs twice, with the answers reversed the second time. Average the two mapped scores. Grade the 311 clear labels; inspect the 85 ties separately.
6. **Read the result.** The judge matched **252/311 (81.0%)**, versus **238/311 (76.5%)** before training. Its probability error, called log-loss, fell from **1.169 to 0.450**. Lower means fewer or less confident mistakes. Shorter-winner accuracy also rose from **59.7% to 68.1%**.
7. **Make the next decision.** Our preset probability-quality rule passed, so a longer 2,000-pair experiment is worthwhile. The accuracy test narrowly missed its cutoff, and tie handling is still weak. These are development results, not final-test performance or calibrated confidence.

**This four-step milestone is finished. The longer run has not started.** The measured next step is to plan its budget and schedule using the corrected 0.00002 rate. Calibrating confidence, evaluating the final test, and connecting the judge to the product remain later work.

[Full results](<../model selection/CORRECTED_JUDGE_COLAB_RESULTS.md>) · [saved evidence and helper guides](<../model selection/results/colab_validation_20261007T224541Z/README.md>). Original checkpoints and optimizer states remain on Drive.

## Next: one full-data A100 run

The new notebook is now running on an A100 with 40 GB of GPU memory. Setup, all 17 tests and the full-model GPU probe passed. Training is in progress; the new accuracy result is not measured yet.

1. Upload the new notebook and package into Colab and choose an **A100 GPU**.
2. Install the separate training environment, run checks, and connect Drive.
3. A short GPU probe checks that the full model fits and saved training state restarts correctly.
4. Train one fresh judge continuously on **all 4,646 eligible clear pairs**. Both answer orders are included: 9,292 presentations, grouped into **1,162 updates**. One epoch means one complete pass; no separate training chunks are needed.
5. Every 100 updates, save progress and check **400 separate monitor pairs**. These pairs never train the model. At the end, keep both the final model and the saved version with the lowest monitor probability error.
6. Compare that selected judge with unchanged Qwen on the same A100 using all 396 validation pairs. Grade 311 clear preferences and inspect 85 ties separately.
7. Read the saved results before deciding the next experiment. If an earlier saved version wins, its report says how much training it actually saw.

The compute sequence has a six-hour maximum; this is a limit, not a speed prediction. Seventeen local and Colab tests passed. Automatic follow-up checks are active. The last actual judge accuracy remains **81.0%** from the earlier L4 experiment. [New notebook, upload ZIP and simple instructions](<../model selection/A100_FULL_DATA.md>). This trains the judge only; final calibration, final-test scoring and product integration remain later work.
