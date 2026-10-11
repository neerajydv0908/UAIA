# Step 8: Verify evidence and applicable outcomes

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/08_verification/METHODS.md)

## My preferred method

Exact complete-passage matching against the cited saved source. Missing citations and changed wording remain unknown.

## Model choice

None for exact checking. Later: compare a passage-support NLI model with an evidence-aware LLM; choose only after independent review.

## Why prefer this now?

We can make a precise, testable claim: these words match this stored passage. This does not establish relevance, source authority, safety, or world truth.

**Why the alternatives wait:** An LLM verifier can interpret paraphrases but introduces another source of mistakes. Arithmetic and isolated code checks belong to separate task-specific extensions.

## What is actually implemented?

Quotation/citation checks implemented. General factual verification, contradiction reasoning, source-authority checks, math/code execution, and safety classification are not implemented.

[Read the relevant code](../../../answerlab/checks.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

PDE-Agents distinguishes execution completion from correctness; AdaRubric motivates keeping central failures visible. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Changed ranges, missing citations, partial quotes, zero claims, and source-version changes are covered.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

