# Step 13: Learn from feedback

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/13_feedback_and_learning/METHODS.md)

## My preferred method

Explicit local feedback storage with separate storage/training consent, pending human review, and no automatic model updates.

## Model choice

None yet. Learning order: a small feature scorer from random weights; an adapted text judge; a separately scoped educational language model pretrained from scratch.

## Why prefer this now?

A user preference can concern length or tone rather than factuality. Reviewing labels before training prevents us from treating every vote as truth.

**Why the alternatives wait:** Immediate online updates and bandit routing are deferred. General preference data alone cannot teach source-conditioned correctness.

## What is actually implemented?

Feedback recording implemented and tested. Human-review tooling, dataset export, training, evaluation releases, and rollback infrastructure are pending.

[Read the relevant code](../../../answerlab/feedback.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

AdaRubric inspires reviewed preference data; Judge Anything motivates independent judge evaluation. DPO generator training is not silently substituted for judge training. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

No consent means no feedback file. Even training consent leaves examples ineligible until review. Saves do not overwrite prior run files.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

