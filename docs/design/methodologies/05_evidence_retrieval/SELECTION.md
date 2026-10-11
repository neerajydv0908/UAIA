# 5. Retrieve authoritative evidence — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

A small saved document collection, human-known relevant passages, then keyword search. Next add one web search service for questions that need it. Keep the evidence package shared across candidates in the main comparison.

Choose dense or hybrid retrieval only after identifying questions the simpler retriever misses.

## Comparison experiment

Compare no evidence, human-selected relevant evidence, and automatically retrieved evidence. The gap between the latter two reveals a retrieval problem; failure even with the correct passage reveals an evidence-use problem.

Measure whether the required passage appears in the first k results, where k is our chosen shortlist size. Also measure final answer correctness, citation support, irrelevant context, cost, and time. Use a frozen document collection for this comparison.

Test near-duplicate pages, conflicting versions, wrong units, outdated sources, absent evidence, and malicious instructions inside a page. Do not reward a system merely for retrieving more text.

## Failure behavior

Record inaccessible, missing, stale, or conflicting evidence separately. Answer only the supported portion, explain uncertainty, or abstain.

**Implementation update:** Local retrieval and the official Python MCP SDK client/server are implemented and integration-tested. Web search/fetching is pending. See [our chosen approach](<../../my_picks/Step_5.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
