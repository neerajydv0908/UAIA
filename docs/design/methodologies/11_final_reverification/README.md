# 11. Reverify the outgoing answer

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It checks the actual text the user will receive. A previous answer may have passed checks, but a rewrite can remove a condition, alter a number, or attach the wrong citation.

**Input:** proposed final text, claim/source mappings, prior verification records, and checker versions.

**Output:** final check result tied to that exact text and evidence version; then release, bounded repair, narrower answer, or abstention.

## Simple example

A source says “up to 40°C when dry.” Synthesis writes “safe up to 40°C in all conditions.” Reusing the old claim's pass result would miss the new error.

Even an unchanged selected answer needs final citation and completeness checks. Some earlier check results can be reused only if their exact inputs and assumptions still match.

## Connection to the papers

PDE-Agents motivates checking outcomes against inputs; AdaRubric motivates essential requirements. Checking the exact outgoing version and invalidating changed records are our engineering controls, not a reproduced paper guarantee.

