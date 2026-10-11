# Step 2: Decide which information is needed

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/02_freshness_decision/METHODS.md)

## My preferred method

Explicit source rules: saved documents, fresh web, no retrieval, or uncertain.

## Model choice

None. Share the task record instead of paying for another classification call.

## Why prefer this now?

The decision and its reason remain visible. The program can refuse to present old saved material as current web evidence.

**Why the alternatives wait:** Always-search is wasteful; never-search fails changing questions. A trained freshness classifier needs reviewed examples.

## What is actually implemented?

Implemented as basic rules. Fresh-web requests currently produce a clear abstention because the web retriever is still pending.

[Read the relevant code](../../../answerlab/steps.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Activation policies inspire selective spending; they do not supply a universal freshness classifier. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Current-fact requests send no generator calls when fresh retrieval is unavailable.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

