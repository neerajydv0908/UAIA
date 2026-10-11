# 1. Understand the question

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It identifies the job the user wants done before we spend money asking models. A question may need several labels: “compare these laptops using current prices” is both comparison and current factual research.

**Input:** the question and only the conversation context needed to understand it.

**Output:** a small task record: task types, requested format, language, important constraints, ambiguity, risk, and a suggested checklist. A checklist is a rubric, not a final verdict.

## Simple example

“What does Device Z's manual say about its operating temperature?” becomes: document question; evidence required; preserve units; answer concisely. If several Device Z manuals are available, the version may need clarification.

Do not rewrite “operating” as “storage” or infer missing facts. Allow “unknown task” rather than forcing a confident category.

## Connection to the papers

AdaRubric motivates matching the checklist to the task. We propose understandable task templates first; automatically generating a fresh rubric for every request is a later experiment. See [paper notes](../../../research_papers.md).

This step identifies the need; step 2 makes the source/freshness decision, and step 3 chooses generators.

