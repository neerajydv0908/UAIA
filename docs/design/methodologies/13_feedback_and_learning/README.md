# 13. Learn from reviewed feedback

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It improves later versions using permitted, reviewed examples. User feedback is a clue about an experience; it is not automatically a factual label.

**Input:** optional feedback, permitted question/answer/source records, checker results, and human review.

**Output:** a versioned dataset, a proposed model/prompt/router update, independent evaluation results, and a release or rejection decision.

## Simple example

A user dislikes a correct answer because it is too long. Record a presentation preference without relabeling its facts as false. A correction with a verifiable source can become a candidate grounding example after review.

“Learning over time” means a controlled offline cycle, not changing weights immediately after a thumbs-up.

## Connection to the papers

AdaRubric shows a path from evaluation to preference data, but its generator-training objective is distinct from our judge. Judge Anything motivates independent judge testing. Activation policies motivate later routing improvements using outcomes. See [paper notes](../../../research_papers.md).

