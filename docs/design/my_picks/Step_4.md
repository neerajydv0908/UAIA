# Step 4: Call providers concurrently

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/04_concurrent_requests/METHODS.md)

## My preferred method

Python asyncio and HTTPX, independent provider outcomes, bounded calls/time, and no automatic retries.

## Model choice

The generators chosen in Step 3. This step has no extra LLM.

## Why prefer this now?

Two requests can overlap while a timeout leaves the other result usable. A direct implementation is small enough to read.

**Why the alternatives wait:** Sequential calls remain useful conceptually for debugging. LiteLLM, LangGraph, and worker queues are deferred until a concrete integration/state problem warrants them.

## What is actually implemented?

Implemented and tested with saved examples and mocked HTTP. Live provider timing/billing has not been measured.

[Read the relevant code](../../../answerlab/providers.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Compare complete request cost and delay, as motivated by activation policies. Parallel calls are our implementation choice. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Tests require two mocked HTTP requests to overlap and verify that a slow fixture does not erase its successful sibling.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

