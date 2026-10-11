# 3. Select suitable answer models — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

A configurable fixed pair after the one-provider prototype. Candidate pair: OpenAI general-purpose tier and Claude middle tier, with exact candidates in the [catalog](../MODEL_CATALOG.md). Compare replacements before adopting them.

This isolates whether comparing answers helps at all. It avoids training a router before we know which outcomes it should predict.

## Comparison experiment

Compare: one strong model, repeated samples from that same model, two different models in parallel, and a cheap-first cascade. Use comparable total spending allowances, including retrieval and judging.

Measure answer correctness, completeness, abstention, cost per request, and typical/p95 waiting time. p95 is the time within which about 95% of requests finish.

Tune escalation using observed checker outcomes and development labels; do not treat a model's “I am 95% confident” as a correctness guarantee. Preserve a final untouched test set.

## Failure behavior

Use a configured replacement when a provider is unavailable and the budget allows it. Record the substitution. If only one candidate arrives, assess it on its own requirements; do not fabricate a comparison.

**Implementation update:** Fixed selection is implemented. OpenAI/Anthropic adapters are tested with simulated HTTP responses; no paid requests have run. Other provider adapters and learned routing are pending. See [our chosen approach](<../../my_picks/Step_3.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
