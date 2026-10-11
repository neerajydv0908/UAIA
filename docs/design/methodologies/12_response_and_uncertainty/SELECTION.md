# 12. Return the answer, citations, and uncertainty — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

A fixed template with nearby citations and plain uncertainty reasons. Show which important claims remain unchecked. Use qualitative statuses such as supported within available evidence, partially supported, conflicting evidence, or insufficient evidence; do not turn them into an unmeasured universal confidence scale.

Show a brief check summary when it helps the user, rather than internal provider logs or hidden system details.

## Comparison experiment

Ask representative users to identify the answer, inspect a source, and explain a stated limitation. Check whether they mistake “supported by this source” for “guaranteed true.”

Measure citation mapping accuracy, comprehension, missing limitations, and task usefulness.

If probabilities are introduced later, use separate calibration data and a final held-out test. Evaluate reliability and error rates alongside answer coverage, and recheck after provider or domain changes.

## Failure behavior

If there is no adequate answer, state the gap and the smallest useful next step. Never invent citations to satisfy the template.

**Implementation update:** CLI and JSON output implemented. Public UI and calibrated probabilities are pending. See [our chosen approach](<../../my_picks/Step_12.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
