# 7. Extract checkable claims

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It turns an answer into statements that can be checked, keeping their meaning and connection to the original text.

**Input:** original candidate answer, question, citations, and task type.

**Output:** claim records with answer spans, claim type, conditions, dates, quantities/units, importance, and cited evidence IDs. A span is the location of text in the original answer.

## Simple example

“Device Z operates at 0–40°C and costs $99 today” contains two claims. They need different evidence. Do not discard “today,” change currency, or lose which device version the statement refers to.

“Device Z looks beautiful” is an opinion. We can assess whether it answers a preference request, but should not invent a factual proof for it.

## Connection to the papers

Task-sensitive judging from AdaRubric and Judge Anything motivates different treatment of factual, creative, and technical content. Our claim-extraction record is an added project design, not a reproduced paper component.

