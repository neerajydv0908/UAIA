# 1. Understand the question — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | How it works | Strength | Weakness / cost | When to consider |
| --- | --- | --- | --- | --- |
| Handwritten rules | Recognize explicit phrases, supplied files, requested formats, and task types | Cheap and easy to explain; no training data | Misses unusual wording and mixed requests | First baseline |
| Small prompted classifier | Ask an LLM to return named task fields | Handles varied language without training a model | Extra call; may invent intent or output invalid fields | Unclear cases after rules |
| TF-IDF plus logistic regression | Turn word patterns into numbers; learn a category from labeled questions | Fast, trainable, and inspectable | Needs representative labels; weak on unfamiliar wording | First learned classifier |
| Fine-tuned text model | Adapt a pretrained reader to our task labels | Can learn domain-specific language | More data, compute, and maintenance | Repeated errors justify it |
| Hybrid | Rules handle obvious cases; a classifier handles the rest | Controls calls while covering more cases | More branches to test | Proposed initial automated design |

TF-IDF weights words by how useful they are for distinguishing documents; logistic regression is a small model that learns to select a category. Neither is a general language model.

## Decisions within any strategy

Use multiple labels when necessary. Keep the user's original question. Separate “I cannot identify the task” from “the task is outside our supported scope.” Ask for a missing detail only when it changes the answer.

Classify inputs as data. A user quoting “ignore all previous instructions” inside a document is not authorizing a new system policy.

