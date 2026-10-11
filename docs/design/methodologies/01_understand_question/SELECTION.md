# 1. Understand the question — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Rules plus a small prompted classifier only for unclear cases. In the first manual version, we supply the labels ourselves.

Use a short initial set: supplied-document Q&A, stable factual Q&A, current factual Q&A, calculation, code, creative writing, and unclear. These are proposed development categories, not a promise to launch every category.

## Comparison experiment

Give the same reviewed questions to rules alone and the hybrid. Include mixed tasks, paraphrases, missing context, and misleading keywords such as “write a story about today's king.”

Measure wrong task labels, missed ambiguity, unnecessary clarification, added cost, and the effect on final answers. Keep questions from the same family together when splitting data.

Adopt the classifier only if it fixes consequential mistakes within the set cost/time allowance. More labels are not inherently better.

## Failure behavior

If the task remains unclear, request the essential missing detail or use a conservative general route. Do not silently choose specialized tools based on a guess.

**Implementation update:** Implemented for English wording with simple rules. Language detection, broad ambiguity understanding, and an LLM fallback are not implemented. See [our chosen approach](<../../my_picks/Step_1.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
