# 7. Extract checkable claims — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Create manual reference examples, then use one structured extraction prompt. Verify that quoted spans exist in the original answer and inspect whether the paraphrased claim preserves their meaning.

Keep a rule-based sentence splitter as a comparison. Add a second extraction/audit call only if it improves important-claim coverage enough to justify cost.

## Comparison experiment

Use answers containing compound statements, pronouns, negation, ranges, dates, citations, opinions, and code blocks.

Measure important-claim coverage, invented claims, lost qualifiers, incorrect source mappings, cost, and time. Human reviewers should examine both the original answer and extracted records.

Compare extraction methods separately from verification. If a claim was never extracted, the verifier cannot be credited for checking it.

## Failure behavior

If reliable extraction fails, mark verification incomplete and use a limited/manual fallback for development. Do not award a perfect support score because zero claims were extracted.

**Implementation update:** Implemented for the quotation contract only. General atomic-claim extraction from free-form answers is pending. See [our chosen approach](<../../my_picks/Step_7.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
