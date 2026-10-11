# 11. Reverify the outgoing answer — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Check the whole final answer in the first implementation, with reuse limited to exact unchanged, still-valid records. Track its final text version. This is simpler to audit than an early optimization that guesses which edits matter.

If the output renderer later writes new prose, route that text back through this step before sending it.

## Comparison experiment

Begin with supported drafts and introduce controlled edits: remove a caveat, swap a citation, change a unit/date, add a plausible unsupported sentence, or merge incompatible versions.

Measure detected regressions, missed regressions, false alarms, added cost/time, and successful repair rate. Include unchanged answers to measure unnecessary rechecking.

Compare full checking against changed-claim checking only after the full baseline works.

## Failure behavior

Stop at the repair limit or deadline. A failed recheck is visible in the final outcome; it must not be hidden behind a polished answer.

**Implementation update:** Implemented for final quotation text and saved source versions. Broader semantic verification remains pending. See [our chosen approach](<../../my_picks/Step_11.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
