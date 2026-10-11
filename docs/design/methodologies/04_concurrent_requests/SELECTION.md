# 4. Ask models concurrently — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Direct adapters with bounded asyncio calls after we understand one successful API request. Start with two generators and configurable limits. A sequential mode remains useful for debugging.

Do not add a workflow framework just to draw the existing sequence. Add one when a concrete state-management problem appears.

## Comparison experiment

First simulate slow responses, failure, refusal, malformed data, and rate limits without paying providers. Then run a small real comparison once the budget is set.

Check that a successful sibling survives a failure, deadlines are respected, duplicate retries are bounded, and usage is recorded honestly. Compare sequential and concurrent elapsed time with the same workload. Parallel work may reduce waiting; it does not make the underlying calls free.

Use a separate cost estimate when final usage is unavailable. Mark the estimate as such.

## Failure behavior

Continue with usable answers if policy allows; otherwise return a clear temporary failure. Never score an empty timeout result as a poor-quality answer from the model.

**Implementation update:** Implemented and tested with saved examples and mocked HTTP. Live provider timing/billing has not been measured. See [our chosen approach](<../../my_picks/Step_4.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
