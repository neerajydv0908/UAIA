# 10. Select an answer or synthesize one — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

- **Python selection policy:** enough for choosing a candidate or a limited response.
- **Template or span assembly:** candidate for extractive composition; no generative model required.
- **Hosted generator from the [catalog](../MODEL_CATALOG.md):** grounded synthesis trial, compared on this task rather than assumed superior.
- **Qwen3-4B or another local generator:** later local composition experiment if quality and hardware fit.
- **Explicit workflow state:** small Python flow first; [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) if bounded revision branches become difficult to maintain.

None of these is a special “truth synthesis” model. The quality control comes from constrained evidence use, checks, and evaluation. Reuse [step 8's tools](../08_verification/TOOLS_AND_MODELS.md) to assess the generated draft.

