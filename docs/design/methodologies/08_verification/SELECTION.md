# 8. Verify claims, citations, calculations, code, and safety — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

For the proposed first text-Q&A scope: validate citation references, inspect support for important factual claims, and apply direct numeric checks when relevant. Start with manual source labels, then compare an evidence-aware model checker against them.

Code execution is a later isolated experiment. Do not describe code as executed or tested before that infrastructure exists.

## Comparison experiment

Build known cases: a real but irrelevant link, correct quote with wrong interpretation, stale source, conflicting sources, wrong unit, correct arithmetic with wrong formula, incomplete evidence, and instructions embedded in a source.

Measure false passes and false failures separately, plus unknown outcomes and coverage. A false pass on a central claim matters more than a cosmetic miss. Check how errors affect answer selection.

For code, add independent hidden edge cases. For safety, test both harmful requests and harmless ones likely to be incorrectly flagged.

## Failure behavior

Record which checks ran and which did not. If a required check fails or cannot run, narrow the claim, request evidence, or abstain; allow a bounded repair only under the request limits.

**Implementation update:** Quotation/citation checks implemented. General factual verification, contradiction reasoning, source-authority checks, math/code execution, and safety classification are not implemented. See [our chosen approach](<../../my_picks/Step_8.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
