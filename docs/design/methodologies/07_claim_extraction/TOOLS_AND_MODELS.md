# 7. Extract checkable claims — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Tool/model | Role |
| --- | --- |
| Python sentence and pattern rules | Initial segmentation baseline |
| Hosted structured-output model from the [catalog](../MODEL_CATALOG.md) | First automated semantic extractor |
| Pydantic record checks | Validate spans, IDs, required fields, and allowed claim types |
| Qwen3-4B or another evaluated local text model | Later local extraction or fine-tuning trial |
| Human annotation records | Reference for missed/changed claims |

[Structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs) helps constrain format; [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) helps validate it. Neither proves the extracted statement faithfully represents the original.

We need short evidence-linked explanations, not access to providers' private internal reasoning.

