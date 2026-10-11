# Step 3: Select candidate models

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/03_model_selection/METHODS.md)

## My preferred method

A configurable fixed provider pair with call limits and a budget check before network work.

## Model choice

Preferred first live trial: OpenAI gpt-6-sol and Anthropic claude-sonnet-5. The default demonstration uses two explicitly labeled saved examples, not real model responses.

## Why prefer this now?

A fixed pair establishes whether comparison helps before we train a router. Exact model IDs remain configurable and their account availability is unverified.

**Why the alternatives wait:** Gemini and DeepSeek are future comparisons. A cheap-first cascade needs outcome data. Calling all providers immediately increases cost without evidence of added value.

## What is actually implemented?

Fixed selection is implemented. OpenAI/Anthropic adapters are tested with simulated HTTP responses; no paid requests have run. Other provider adapters and learned routing are pending.

[Read the relevant code](../../../answerlab/providers.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

Optimal Model Activation Policies motivates the later fixed-pair versus cascade experiment, not a claim of optimality now. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

A zero estimated-spend allowance blocks live calls before HTTP. The first live configuration contains no invented prices.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

