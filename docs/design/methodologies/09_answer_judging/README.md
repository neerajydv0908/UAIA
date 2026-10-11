# 9. Judge and score the answers

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It compares answers using the user's task, source evidence, and checker results. Its job is broader than truth alone: an answer can be supported but incomplete or fail the requested format.

**Input:** question, shared evidence, candidate answers without provider labels, claim/check records, and a task rubric.

**Output:** eligibility, scores by relevant dimension, comparison outcome, brief evidence-linked reason, and unresolved issues.

## Simple example

Answer A is fluent but gives Device Z's wrong operating range. Answer B is plain but supported. The central factual failure excludes A; style should not reverse the outcome.

Allow A wins, B wins, both adequate/tie, neither adequate, and insufficient evidence. “Both equally bad” is not an adequate tie.

## Connection to the papers

AdaRubric motivates task-specific criteria and minimum requirements. Judge Anything motivates evaluating judges rather than assuming strong writers are strong judges. Our provider blinding and order swaps are additional controls. [Full paper explanations](../../../research_papers.md).

