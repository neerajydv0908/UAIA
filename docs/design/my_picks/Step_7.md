# Step 7: Extract claims

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/07_claim_extraction/METHODS.md)

## My preferred method

An explicit quotation format: one complete source passage and its citation per answer line. Every nonempty line becomes a check record.

## Model choice

None in this teaching baseline. Preferred next semantic-extraction trial: a small structured-output model such as gpt-6-luna, evaluated against human-labeled spans.

## Why prefer this now?

This narrow contract lets us test that nothing is silently dropped. It is easier to understand than pretending a basic sentence splitter understands all claims.

**Why the alternatives wait:** General LLM extraction is deferred until we have qualifier/coverage examples. A trained extractor needs labeled source spans.

## What is actually implemented?

Implemented for the quotation contract only. General atomic-claim extraction from free-form answers is pending.

[Read the relevant code](../../../answerlab/checks.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Task-sensitive judging motivates preserving qualifiers; this strict extraction contract is a project baseline. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Uncited extra prose remains unchecked and makes the candidate ineligible. Removing a condition does not pass.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

