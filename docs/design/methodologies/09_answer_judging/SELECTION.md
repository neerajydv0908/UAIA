# 9. Judge and score the answers — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Required verifier checks plus one evidence-aware pairwise judge with a short task template. Hide provider identity, include ties/neither outcomes, and request a brief reason referencing evidence IDs.

Use order swaps during development to quantify bias. In a later service, decide whether every request or only a sampled/uncertain subset merits the extra judging cost.

## Comparison experiment

Compare a generic “choose the best” prompt, a fixed task rubric with evidence, and an adaptive rubric. Hold candidates and sources fixed; report total cost. Add an existing open reward baseline and our learned model only after data exists.

Measure agreement with independent reviewers, central-error false passes, ties/neither accuracy, answer-order stability, sensitivity to irrelevant verbosity, cost, and time.

Change fictional evidence while keeping answers fixed. A source-aware judge should change its choice appropriately. Also test absent evidence, conflicting sources, and candidate text demanding a high score.

## Adoption conditions

A learned judge must earn its place through quality, cost, or speed without unacceptable error-category regressions. Validate on new question/source groups and later dates. Keep final test data out of prompt/weight selection.

**Implementation update:** Rule eligibility and ranking implemented. Prompted and trained judges, semantic relevance/completeness checks, and calibrated scores are pending. See [our chosen approach](<../../my_picks/Step_9.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
