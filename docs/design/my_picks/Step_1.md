# Step 1: Understand the question

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/01_understand_question/METHODS.md)

## My preferred method

Readable rules with multiple task labels and an explicit clarification flag.

## Model choice

None in the working baseline. A small prompted classifier such as gpt-6-luna is a later comparison, not a hidden dependency.

## Why prefer this now?

We can inspect exactly which phrase caused a label. Training a classifier before collecting examples would make the first version harder without demonstrated benefit.

**Why the alternatives wait:** Always calling an LLM adds cost. TF-IDF/logistic regression needs labeled questions. Both remain experiments when the rules show consequential failures.

## What is actually implemented?

Implemented for English wording with simple rules. Language detection, broad ambiguity understanding, and an LLM fallback are not implemented.

[Read the relevant code](../../../answerlab/steps.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

AdaRubric motivates choosing task-appropriate checks; our rules are an engineering baseline. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Mixed document/comparison requests, ambiguous references, blank input, and creative requests containing “today” are tested.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

