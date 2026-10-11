# 10. Select an answer or synthesize one — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Return an adequate original answer. Introduce synthesis only after we have examples where useful, compatible information is split across candidates.

If two answers are equally adequate, a documented tie-break such as completeness then concision can choose the display version. It must not pretend one is more factually reliable.

## Comparison experiment

Hold candidates and evidence fixed. Compare original selection, extractive assembly, and grounded synthesis. Have reviewers judge final usefulness without knowing the method.

Measure gained completeness, unsupported new claims, lost qualifications, citation correctness, cost, and time. Include conflicting sources and cases where a short original already fully answers the question.

Adopt synthesis only when its usefulness gain outweighs measured regressions under set product limits.

## Failure behavior

If no answer meets central requirements, use a bounded retrieval/escalation path or return a supported partial answer/abstention. Do not always pick the highest number.

**Implementation update:** Original-answer selection and abstention implemented. Synthesis and debate are not implemented. See [our chosen approach](<../../my_picks/Step_10.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
