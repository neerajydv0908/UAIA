# Source register and research boundaries

Compiled: 2026-09-28. Official pages and maintainer model cards below informed the methodology options. Current documentation supports capabilities and names; it does not establish which option wins on our product. Follow links in each step for the relevant claim.

## The supplied papers

The local PDFs were reviewed on 2026-09-26; this library uses the detailed [paper notes](../../research_papers.md). We have not independently reproduced the papers' experiments.

| Short name | Local original | External record | Main use here |
| --- | --- | --- | --- |
| AdaRubric | kept locally | [arXiv](https://arxiv.org/abs/2603.21362) | Task checklists and keeping serious dimension failures visible |
| Activation policies | kept locally | [arXiv](https://arxiv.org/abs/2609.15992) | Measured escalation under cost constraints |
| PDE-Agents | kept locally | [arXiv](https://arxiv.org/abs/2606.07850) | Evidence provision and checking inputs as well as completion |
| Judge Anything | kept locally | [Authors' project](https://urrealhero.github.io/judgeanythingweb/) | Evaluate the judge; compare pairwise and scoring protocols |

## Provider and integration documentation

- [OpenAI model catalog](https://developers.openai.com/api/docs/models), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [embeddings](https://developers.openai.com/api/docs/guides/embeddings), [moderation](https://developers.openai.com/api/docs/guides/moderation).
- [Claude model catalog](https://platform.claude.com/docs/en/models/overview), [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models), [DeepSeek API](https://api-docs.deepseek.com/).
- [Perplexity overview](https://docs.perplexity.ai/docs/getting-started/overview) and [Search API](https://docs.perplexity.ai/docs/search/quickstart).
- [MCP architecture](https://modelcontextprotocol.io/docs/learn/architecture): tool connections, not an answer-ranking algorithm.
- [Python asynchronous tasks](https://docs.python.org/3/library/asyncio-task.html), [HTTPX async client](https://www.python-httpx.org/async/), [LiteLLM](https://docs.litellm.ai/docs/), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview).
- [Pydantic validation](https://pydantic.dev/docs/validation/latest/get-started/).

## Retrieval and checking

- [SQLite FTS5](https://www.sqlite.org/fts5.html), [Sentence Transformers retrieval/reranking](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html), [pgvector](https://github.com/pgvector/pgvector).
- [Tavily](https://docs.tavily.com/documentation/about), [Trafilatura text extraction](https://trafilatura.readthedocs.io/en/latest/).
- [SymPy](https://docs.sympy.org/latest/index.html), [pytest](https://docs.pytest.org/en/stable/), [E2B sandbox documentation](https://docs.e2b.dev/).
- Specific embedding, NLI, and reward-model maintainer cards are linked in the [model catalog](MODEL_CATALOG.md).

## Learning and evaluation tools

- [scikit-learn text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) and [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [PyTorch beginner learning guide](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
- [TRL reward modeling](https://huggingface.co/docs/trl/reward_trainer).
- [Label Studio](https://labelstud.io/guide/).
- Dataset sources, splits, and limitations were reviewed on 2026-09-26 and are maintained in [DATASETS.md](../../datasets.md), rather than treated as newly downloaded or commercially cleared here.

## How to interpret recommendations

“Paper-inspired” describes a transfer of an idea, not a replication. “Proposed” describes our design judgment. “Documented” describes a maintainer's stated capability. “Measured” would require a recorded project experiment; none has run yet.

Recheck prices, model access, data permissions, and exact library versions when implementing. We do not infer commercial training permission merely from an available download or a model-card badge.
