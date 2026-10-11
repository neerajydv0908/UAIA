# 12. Return the answer, citations, and uncertainty

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It turns the checked result into a clear user-facing response. The user needs the answer first, useful source links, and a plain explanation of important limitations.

**Input:** checked final text, claim/source links, uncertainty reasons, and check coverage.

**Output:** an answer or clearly limited result that preserves the checked meaning.

## Simple example

For our invented manual: “Device Z's operating range is 0–40°C according to the supplied manual. I could not verify whether a newer version changes that range.”

This is more informative than “95% confident” when we have not measured what that percentage means.

## Connection to the papers

Judge Anything distinguishes judge agreement from answer correctness. AdaRubric's internal confidence is not automatically calibrated for users. Our display policy must reflect the evidence and measured reliability we actually have.

