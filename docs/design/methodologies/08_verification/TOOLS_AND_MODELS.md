# 8. Verify claims, citations, calculations, code, and safety — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Tool/model | Proposed use | Important limit |
| --- | --- | --- |
| Python arithmetic/Decimal with restricted inputs | Recalculate numeric claims | Still check units and formula choice |
| SymPy | Symbolic calculations where appropriate | Validate assumptions and safe parsing |
| pytest in a sandbox | Run independent code tests | pytest itself is not a sandbox |
| E2B or a separately reviewed isolated execution service | Candidate environment for code evaluation | Must configure limits and verify isolation; later milestone |
| DeBERTa NLI model from the [catalog](../MODEL_CATALOG.md) | Passage-to-claim support baseline | Test dates, quantities, domain mismatch, and truncation |
| Evidence-aware prompted LLM | More flexible source interpretation | Fallible and potentially biased |
| OpenAI moderation API | One content-policy classifier option | Does not replace tool isolation or factual checks |

Primary sources: [SymPy](https://docs.sympy.org/latest/index.html), [pytest](https://docs.pytest.org/en/stable/), [E2B](https://docs.e2b.dev/), [NLI model card](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli), [moderation](https://developers.openai.com/api/docs/guides/moderation).

For training source support, compare [RAGTruth, FEVER, and SciFact](../../../datasets.md); their labels and domains differ.

