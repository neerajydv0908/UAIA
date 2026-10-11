# 2. Decide whether fresh information is needed — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Use explicit rules plus an uncertain outcome. Distinguish “no external evidence needed,” “supplied/saved source needed,” and “fresh search needed.” On factual uncertainty, attempt suitable retrieval within the budget; if unavailable, keep the limitation visible.

A cache policy comes after reliable source records exist.

## Comparison experiment

Label a mix of stable, changing, historical, supplied-document, and creative questions. Compare always-search, rule-based decisions, and a prompted classifier. Freeze the evaluation date or specify the date required by each question.

Count missed necessary searches separately from unnecessary searches. Also measure stale-answer errors, retrieved-source relevance, search spending, and added time.

Choose thresholds on development questions. For topics with a high cost of being outdated, prioritize avoiding stale answers over saving a small search fee.

## Failure behavior

A failed search is not “search unnecessary.” A missing publication date is not “current.” Return a dated limitation, a narrower supported answer, or clarification when the source need cannot be satisfied.

**Implementation update:** Implemented as basic rules. Fresh-web requests currently produce a clear abstention because the web retriever is still pending. See [our chosen approach](<../../my_picks/Step_2.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
