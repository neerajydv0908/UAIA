# 6. Convert responses to a common format — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | How it works | Advantage | Limitation |
| --- | --- | --- | --- |
| Plain dictionaries | Manually assemble named fields | Simplest starting point | Mistyped or missing fields can spread |
| Dataclasses with explicit checks | Define a Python record and validate intentionally | Lightweight and readable | Validation requires additional care |
| Pydantic models | Define allowed fields/types and validate inputs | Clear errors at component boundaries | A correct shape does not mean correct content |
| Provider-native structured output | Ask a supported model for a constrained structure | Useful for claim/classification results | Does not replace the provider adapter or semantic checks |
| LLM-based repair | Ask another call to fix malformed generated structure | Sometimes recovers a result | Can change meaning and adds cost; exceptional fallback |

A schema is the blueprint describing expected fields. Schema validation asks “is this well formed?” Factual verification asks “is this supported?” They solve different problems.

## Preserve rather than erase differences

Keep finish reasons, refusals, truncation, tool/search use, and model substitutions visible. Store provider-reported usage separately from our estimated cost. Judge inputs can hide branding while internal provenance keeps it.

If an answer exceeds a context limit, record an explicit truncation policy. Silently dropping its ending can change the comparison.

