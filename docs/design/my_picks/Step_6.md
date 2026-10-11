# Step 6: Normalize provider responses

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/06_response_normalization/METHODS.md)

## My preferred method

Explicit provider adapters and Pydantic records, preserving original text, returned model ID, finish status, usage, and timing.

## Model choice

None. Known fields are mapped by code rather than rewritten by a model.

## Why prefer this now?

Missing usage stays unknown and truncation stays distinct from a complete answer. We can validate the record before another component trusts it.

**Why the alternatives wait:** Plain dictionaries are useful for learning but lack the chosen boundary validation. LLM repair and gateway normalization add unnecessary interpretation at this stage.

## What is actually implemented?

Implemented for the two current adapters and fixtures. Raw provider payloads are not automatically retained.

[Read the relevant code](../../../answerlab/records.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Reproducible comparisons need consistent records; this schema is our own engineering choice. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Tests cover multiple text blocks, missing usage, empty success, refusals, truncation, and returned model identity.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

