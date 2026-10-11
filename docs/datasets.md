# Datasets for our judge and our learning models

Sources originally checked: 2026-09-30. Audit updated: 2026-10-05. **Current experimental release: 8,563 HelpSteer2 preference pairs in v2**, after 558 further rows were held aside for text overlap or unresolved sample review. Twenty fictional practice examples had a model re-review and still await independent human review; they remain ineligible for training. No model has been trained; other datasets below remain recommendations.

## What a dataset is

A dataset is a collection of examples. A **label** tells the model what it should learn from an example: which answer is better, whether a claim is supported, or whether code passes specified tests.

We need different examples for different jobs. General text teaches language. Preference examples teach comparison. Evidence-linked examples teach checking against sources. A test set measures learning on examples we did not use to train or choose settings.

## Start with these

| Dataset | What it contains and teaches | How we would use it | Important limitation |
| --- | --- | --- | --- |
| [HelpSteer2 / HelpSteer2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2) | Human ratings of answers; the preference release adds direct answer comparisons and explanations | First public data for general answer judging; inspect the preference files for pairwise training | Does not provide a trusted evidence package for every answer. Verbosity and complexity are descriptive attributes, not qualities to maximize. Card lists CC BY 4.0. |
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | Nearly 18,000 generated responses with manually marked unsupported/contradictory spans and source records | Grounding research reference; QA portion excluded from the product training pool pending underlying permissions | Span labels are not whole-answer preference labels. The repository lists MIT, but its QA source, [MS MARCO](https://microsoft.github.io/msmarco/), specifies non-commercial research. Other source collections need separate review. |
| Our own reviewed comparisons | Questions from the intended users, candidate answers, evidence, checks, and human decisions | Main data for the actual product's evidence-conditioned judge | Start small and inspect quality. Raw thumbs-up feedback is not a correctness label. |

[HelpSteer3-Preference](https://huggingface.co/datasets/nvidia/HelpSteer3#preference) is also a current candidate: the card describes 40,476 preference examples and CC BY 4.0. We started with the smaller HelpSteer2 preference download for a manageable first preparation pass. HelpSteer3 is not downloaded or assumed superior on our task. Check dataset overlap before mixing versions.

HelpSteer2 and its preference extension are in the same repository but use different record structures. The preference download includes an internal split field; do not assume everything loaded under an outer `train` name is training data. Honor the documented split and group related prompts together. [Maintainer's format instructions](https://huggingface.co/datasets/nvidia/HelpSteer2).

## Add these only for a specific gap

| Dataset | Role in our project | Caution |
| --- | --- | --- |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Optional broad preference training: about 64,000 prompts, four responses each, and GPT-4-generated feedback | Synthetic judgments can transfer the annotating model's biases. The card lists MIT but identifies several upstream datasets. It includes TruthfulQA prompts, so training on it can contaminate a later TruthfulQA evaluation. |
| [FEVER](https://fever.ai/dataset/fever.html) | Claim verification: supported, refuted, or not enough information relative to a Wikipedia evidence corpus | Its historical corpus is not live-web truth. Train and evaluate with the correct evidence snapshot and official splits. |
| [SciFact](https://github.com/allenai/scifact) | Scientific claims with evidence sentences and support/refutation labels; useful for learning claim-to-passage checks | Small, specialized domain. Public test labels are not provided in the repository; use documented evaluation procedures and do not repurpose the test set as training data. |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Educational pretraining of a tiny language model from random weights | Synthetic story text is not factual verification or reward-model supervision. Card lists CDLA Sharing 1.0. It is a learning resource, not a ready-made product judge dataset. |

## Reserve these for evaluation

| Dataset / suite | What it tests | Our rule |
| --- | --- | --- |
| [RewardBench 2](https://huggingface.co/datasets/allenai/reward-bench-2) | Reward-model judgments across factuality, instruction following, math, safety, focus, and ties | Evaluation only in this project. Use its official multi-answer protocol; reducing it to arbitrary pairs changes the benchmark. Card lists ODC-BY, research/educational intent, and separate third-party output terms. Review the intended use before commercial integration. |
| [EvalPlus: HumanEval+ and MBPP+](https://github.com/evalplus/evalplus) | Generated code against expanded test suites | Keep benchmark tests out of development/training feedback. Execute candidate code only in an isolated environment. Passing tests does not establish complete correctness. |
| [GSM8K](https://github.com/openai/grade-school-math) | Grade-school math word problems with worked answers | Use the test split only for evaluation; use train data separately if teaching math verification. Add fresh problems because public benchmark exposure is possible. |
| [JudgeAnything](https://urrealhero.github.io/judgeanythingweb/) | Judge behavior across modalities | Later evaluation if we add images, audio, or video; keep separate from training. |

Public benchmarks are useful diagnostics. Our central product test must also contain new questions reflecting the users we intend to serve.

## Our own record format

We will agree on the format before collecting data. A record should keep:

| Part | Fields in ordinary language |
| --- | --- |
| Question | ID, text, task type, date, language, and permission to retain/use it |
| Evidence | Passage ID, exact saved text or authorized snapshot reference, URL, publisher, publication date if known, retrieval time, and content hash |
| Candidates | Original answer text, provider/model version, generation settings, usage, latency, and failure status |
| Claims | Exact answer spans, preserved conditions, cited passage IDs, and support/contradiction/unknown labels |
| Checks | What ran, its version, outcome, and its limitations; use separate values for pass/fail/unknown/not applicable |
| Human judgment | A wins / B wins / both adequate / neither adequate / insufficient evidence, plus a brief evidence-based reason |
| Data management | Annotator agreement, split/group IDs, source license, provenance, revision, and permitted uses |

Two adequate answers can tie. Two incorrect answers must not be labeled a useful tie just because their scores match. Human preference and factual correctness are stored separately.

## A small collection plan

1. Begin with 20–30 manually reviewed examples to clarify what labels mean. This is practice data, not a credible final benchmark.
2. Assemble a few hundred development questions covering stable facts, current facts, provided documents, conflicting evidence, and unanswerable requests. Narrow these categories to the chosen launch scope.
3. Obtain at least two independent reviews on a useful subset. Resolve disagreements or preserve uncertainty; do not manufacture a winner.
4. Build useful difficult examples: a wrong date, changed unit, fabricated citation, fluent unsupported answer, two equivalent answers, and a source that contradicts model memory. Mix controlled edits with real model mistakes so the judge cannot learn only our editing patterns.
5. Create a separate fresh test collection before selecting the final approach. Keep related questions, paraphrases, document versions, and candidate answers in the same split.
6. Expand training in measured increments. A few hundred labels can teach us about the pipeline but do not guarantee a strong text judge. Use a learning curve: plot performance as we add reviewed examples.

## Teaching evidence-conditioned comparison

Give the judge the same question and evidence package for both candidates. Train claim support first or jointly with answer comparison. For a preferred answer, increase its score relative to the worse answer; keep ties and unanswerable cases explicitly represented. Later, fit confidence on a separate calibration set.

To test whether the model actually reads evidence, change the evidence while keeping the candidate text fixed, using clearly fictional facts or documented versions. Its supported choice should change appropriately. Also include irrelevant and missing evidence. Do not teach that a higher citation count means a better answer.

HelpSteer2's ratings, UltraFeedback's feedback, and RAGTruth's spans mean different things. We will define an explicit conversion for each source, retain its original labels, and avoid pretending all numerical scales measure the same quantity.

## Before using data in training

For each dataset revision, record its source, license, attribution requirements, upstream content conditions, provider-output restrictions, intended use, and overlap with our test sets. Verify API terms before using collected provider outputs to train a model; do not assume all providers permit every training use. Store only user data that has the required permission and redact secrets. No dataset is approved for a commercial model merely because it is downloadable.
