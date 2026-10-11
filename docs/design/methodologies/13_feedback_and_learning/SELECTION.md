# 13. Learn from reviewed feedback — selection plan

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Proposed starting choice

Start with optional feedback collection and manual review. Improve clear pipeline failures before training. Then train a small feature scorer as the first ML experiment and compare a pretrained text-judge adaptation as the practical product experiment.

Keep scratch language-model training as a separate learning experiment with its own measured compute plan. It enters the product only if it passes the same evaluation.

## Comparison experiment

Compare current rules/prompted judge, a small learned scorer, an existing open reward model, and our adapted text judge on the same candidate/evidence records.

Use grouped training, validation, calibration, and test splits. Track preference accuracy, central-error false passes, grounding behavior, inference cost, waiting time, and results by task type. Plot performance as reviewed training examples increase.

Use evidence-change tests: keep candidate text fixed, change a fictional source fact, and check whether the judge responds to the evidence. Include missing and irrelevant sources. Retain ties and neither-adequate labels.

## Release and data decisions

Human-review a useful subset independently; preserve disagreement. Record permitted training uses, source provenance, provider-output terms, consent, and retention. Removing a stored row does not automatically remove its influence from already-trained weights; versioning must support the chosen data policy.

Release only after the predefined quality/cost criteria pass; retain the previous version. Online self-updating is deferred.

**Implementation update:** Feedback recording implemented and tested. Human-review tooling, dataset export, training, evaluation releases, and rollback infrastructure are pending. See [our chosen approach](<../../my_picks/Step_13.md>) for the current decision and [BUILD_STATUS.md](../../../build_status.md) for validation.
