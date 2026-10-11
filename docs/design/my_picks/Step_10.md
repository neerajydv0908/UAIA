# Step 10: Select or synthesize

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/10_selection_and_synthesis/METHODS.md)

## My preferred method

Select an eligible original answer; break equal heuristic scores by concision and a stable tie-break while preserving the tie record.

## Model choice

None. If grounded synthesis later earns adoption, trial the existing generator pool rather than add another model without evidence.

## Why prefer this now?

Keeping the original avoids a new generation step that can invent claims or drop conditions.

**Why the alternatives wait:** Extractive combination and model synthesis may improve completeness, but must demonstrate benefits against selection under the same budget.

## What is actually implemented?

Original-answer selection and abstention implemented. Synthesis and debate are not implemented.

[Read the relevant code](../../../answerlab/checks.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Judge Anything motivates distinguishing relative preference from absolute adequacy. Synthesis remains our proposed experiment. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Only completed candidates whose every extracted line passes quotation checking can be selected.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

