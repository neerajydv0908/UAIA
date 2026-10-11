# Step 12: Present the result and uncertainty

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/12_response_and_uncertainty/METHODS.md)

## My preferred method

A fixed CLI presentation with source references, an optional 13-step trace, and explicit uncertainty reasons.

## Model choice

None. Rendering does not invoke a rewriting model.

## Why prefer this now?

The displayed answer remains the checked text. We avoid unmeasured confidence percentages and expose the narrow scope of the check.

**Why the alternatives wait:** A web interface, per-claim inspection, and calibrated probabilities are useful later product work. They are not prerequisites for understanding the pipeline.

## What is actually implemented?

CLI and JSON output implemented. Public UI and calibrated probabilities are pending.

[Read the relevant code](../../../answerlab/pipeline.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Judge Anything's lesson is that agreement and correctness differ; our UI does not equate them. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

The demo labels saved fixtures and fictional sources, displays actual checking limits, and distinguishes unknown costs.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

