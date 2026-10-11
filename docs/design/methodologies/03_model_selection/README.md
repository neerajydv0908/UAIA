# 3. Select suitable answer models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

The router chooses which models to call and whether another attempt is worth its cost. A router can begin as a simple list; it does not need to be a trained model.

**Input:** task and evidence requirements, model capabilities, availability, privacy constraints, deadline, and spending allowance.

**Output:** selected models, generation settings, and any permitted escalation rule. Reserve part of the request budget for checking and final presentation.

## Simple example

During the baseline experiment, every supported question goes to the same two configured providers. Later, if the first answer fails a central check, a second model may be called if time and budget remain.

The question “which models are best?” is answered using our own task results. Brand popularity and agreement between providers do not establish accuracy.

## Connection to the papers

Optimal Model Activation Policies motivates a cheap-first two-model cascade and measured thresholds. Its assumptions and scope do not prove our multi-provider router optimal. [Detailed review](../../../research_papers.md).

