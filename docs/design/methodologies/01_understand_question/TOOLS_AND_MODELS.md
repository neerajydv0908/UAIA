# 1. Understand the question — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Tool/model | What we would use it for | First-version fit |
| --- | --- | --- |
| Python rules and a small label list | Transparent categorization | Start here |
| Pydantic | Validate the task record's allowed fields | Useful once the record is coded |
| Small hosted model from the [catalog](../MODEL_CATALOG.md) | Structured classification of ambiguous wording | Compare cost and mistakes before selecting |
| scikit-learn TF-IDF + LogisticRegression | Train a compact classifier using our labels | Later educational ML baseline |
| Qwen3-4B or another tested local text model | Local prompted/fine-tuned classifier | Requires measured hardware feasibility |

Primary references: [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/), [text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression). Specific model cards are in the catalog.

A valid record does not mean the chosen category is correct. No model training is needed for the rule baseline.

