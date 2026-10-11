# Models by job, not by popularity

Documentation checked: 2026-09-28. **These are candidates for evaluation, not measured project winners.** API availability, aliases, pricing, and supported parameters can change. Verify account access and pin the returned model/version when implementation begins.

## Hosted answer generators and possible prompted judges

A **prompted judge** is an ordinary language model given a question, evidence, candidate answers, and a scoring checklist. We must test its judging separately from its writing.

| Provider | Concrete documented candidates | Possible role in our experiments | Source |
| --- | --- | --- | --- |
| OpenAI | `gpt-6-sol`; `gpt-6-luna`; `gpt-6-astra` | General candidate; lighter classification/extraction trial; stronger comparison trial | [Official model catalog](https://developers.openai.com/api/docs/models) |
| Anthropic | `claude-sonnet-5`; `claude-haiku-4-5-20251001`; `claude-opus-5-5` | Candidate/judge trial; smaller helper trial; more demanding candidate trial | [Official Claude models](https://platform.claude.com/docs/en/models/overview) |
| Google | `gemini-3.8-flash`; `gemini-3.5-flash-lite`; `gemini-3.1-pro-preview` | Stable candidate; lighter helper trial; preview comparison only | [Official Gemini catalog](https://ai.google.dev/gemini-api/docs/models) |
| DeepSeek | `deepseek-flash`; `deepseek-v4-pro` | Additional candidate and cost/quality comparisons | [Official API documentation](https://api-docs.deepseek.com/) |
| Perplexity | Search API; Agent API with an explicitly recorded backing model | Search results for our retrieval pipeline, or a separately labeled search-backed answer | [Official Search guide](https://docs.perplexity.ai/docs/search/quickstart) |

These roles are our hypotheses. A smaller advertised tier is not automatically cheaper for the complete task if it needs more retries. No unsupported cross-provider quality ranking is intended.

For Perplexity, distinguish raw search results from generated answers. Its current documentation places Sonar under a legacy API section; check the current integration path before choosing a Sonar-based adapter. A routed answer may share its underlying model with another candidate, so record the backing model and avoid counting it as an independent vote. [Perplexity API overview](https://docs.perplexity.ai/docs/getting-started/overview).

For an initial pair, trial `gpt-6-sol` and `claude-sonnet-5`, subject to access and a set budget. Trial `gemini-3.8-flash` or DeepSeek as replacements. A provider different from the generators is one possible judge candidate; different branding still does not guarantee independent errors.

## Open models for more specific jobs

“Open weights” means we can obtain the trained model's parameters under its stated conditions. It does not mean zero hosting cost or unrestricted use of every training source.

| Candidate | Job to explore | Limitation to test |
| --- | --- | --- |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Local prompting or starting point for a custom text judge | Already pretrained; adapting it is not scratch pretraining. Hardware feasibility is unmeasured. |
| [Skywork-Reward-V2-Qwen3-4B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B) | Existing scalar reward-model baseline | A reward score is relative preference, not factual correctness. Adding evidence to its input does not prove it uses evidence reliably. |
| [Prometheus 2, 7B](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0) | Rubric-guided judge baseline | Documented grading templates require a reference answer. Do not invent a trusted reference from the candidate being judged. Review its stated data-use conditions. |
| [DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) | Passage-to-claim support classification | NLI means checking whether one text supports, contradicts, or leaves another undecided. It does not inspect the world; long passages, dates, and quantities need testing. |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Local retrieval representation | Similarity finds relevant-looking passages; it does not verify them. |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Reorder a short list of retrieved passages | Adds computation and cannot recover evidence absent from the initial list. |
| [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings), including `text-embedding-3-small` | Hosted embedding alternative | Sends the embedded text to a service; measure relevance and full indexing/query cost. |

The catalog is intentionally a shortlist. Some entries are older, useful baselines rather than the newest release. Inspect each exact model revision, license, input template, supported runtime, and training provenance before adoption.

## Which steps need a model?

Steps 1–2 can begin with rules. Step 3 can be a lookup table. Step 4 is request orchestration. Step 6 is data validation. Step 12 can be a template. Step 13 begins with review and data management. None of these requires training a new LLM just to exist.

Generation, difficult claim extraction, evidence interpretation, and judging are the most useful early places to evaluate language models. Arithmetic, schema validation, and executing independent code tests have more direct tools.

## What belongs in our future model registry

For each role, record the exact provider/model identifier, revision if available, allowed input types, structured-output support, context limits, usage units, price-check date, timeout, rate limits, permitted data uses, and evaluation results. Keep API keys outside this registry.

A registry is simply a configuration list. When an alias changes, rerun representative checks before accepting it as equivalent. Do not assume an “OpenAI-compatible” endpoint has identical capabilities, usage accounting, or error behavior.
