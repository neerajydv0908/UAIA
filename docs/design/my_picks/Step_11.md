# Step 11: Recheck the final text

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/11_final_reverification/METHODS.md)

## My preferred method

Re-extract and check the exact outgoing text, then compare claim/source fingerprints with its prior checks.

## Model choice

None for the implemented quotation scope. An independent final critic remains a later experiment.

## Why prefer this now?

An edited range, missing condition, or changed source version invalidates an old result. We avoid trusting a check for a different answer.

**Why the alternatives wait:** Changed-claim-only checking could reduce future cost but is premature while our checks are inexpensive. Automatic repair loops are deferred.

## What is actually implemented?

Implemented for final quotation text and saved source versions. Broader semantic verification remains pending.

[Read the relevant code](../../../answerlab/checks.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Input fidelity and essential requirements inspire this control; exact-output fingerprints are our design. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Changing a previously checked answer or source version makes final verification fail.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

