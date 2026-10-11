# 8. Verify claims, citations, calculations, code, and safety

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It runs checks matched to the claim. This is several small checking jobs, not one universal truth detector.

**Input:** claims, relevant source passages, task requirements, and allowed verification tools.

**Output:** check records with method/version, evidence, result, and scope. Factual support labels are supported, contradicted, insufficient evidence, or not applicable. Tool checks separately record pass, fail, unknown, or not applicable.

## Simple example

A citation may open successfully while the page says nothing about Device Z's operating range. The link check passes; the support check remains unresolved.

A calculator may confirm 12 × 8 = 96, but it cannot establish that multiplying these quantities answers the user's question.

## Connection to the papers

PDE-Agents illustrates that successful execution can use wrong inputs. AdaRubric motivates keeping central correctness failures visible instead of averaging them away. These lessons guide our checks; they do not establish universal verification. [Paper notes](../../../research_papers.md).

