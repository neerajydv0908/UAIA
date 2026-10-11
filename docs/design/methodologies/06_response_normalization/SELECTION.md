# 6. Convert responses to a common format — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Start with a plain dictionary for the record, then use explicit adapters and Pydantic at component boundaries. Preserve original text and metadata.

Proposed fields include request ID, candidate ID, provider/model version, answer text, status, finish reason, citations, evidence-package ID, latency, usage, cost estimate, and error details without secrets.

## Comparison experiment

Use saved provider-shaped fixtures: successful text, multiple content blocks, refusal, incomplete answer, missing usage, malformed tool result, and an empty response.

Verify round-trip traceability: a reviewer can find the original source for every normalized answer. Check that no status becomes a false success and no number is fabricated.

This is primarily a data-integrity test, not a contest between LLMs.

## Failure behavior

Reject or quarantine an invalid record with its reason. If repair is later added, preserve both versions, label the repair, cap attempts, and revalidate. Never silently overwrite the original answer.

**Implementation update:** Implemented for the two current adapters and fixtures. Raw provider payloads are not automatically retained. See [our chosen approach](<../../my_picks/Step_6.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
