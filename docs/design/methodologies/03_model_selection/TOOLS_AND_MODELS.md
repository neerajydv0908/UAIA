# 3. Select suitable answer models — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

- **Configuration table in Python:** proposed first router; lists allowed providers and task capabilities.
- **Direct provider adapters:** OpenAI, Anthropic, Gemini, DeepSeek, and Perplexity-related options appear in the [model catalog](../MODEL_CATALOG.md).
- **LiteLLM:** an optional common API/gateway layer. It can reduce integration work but does not determine our evidence-quality policy. [Official documentation](https://docs.litellm.ai/docs/).
- **scikit-learn:** later supervised router predicting recorded task outcomes. [Supervised linear models](https://scikit-learn.org/stable/modules/linear_model.html).
- **Our trained judge:** can supply an escalation feature later, after its score is evaluated. It must not certify itself correct.

Keep exact model identity, routing reason, rate-limit status, and estimated/actual cost. A model alias can change over time.

