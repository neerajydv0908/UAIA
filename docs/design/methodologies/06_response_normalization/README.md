# 6. Convert responses to a common format

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It translates provider-specific envelopes into our own consistent records. **Normalization** here means organizing fields, not rewriting answer meaning.

**Input:** the provider's original response and request metadata.

**Output:** one candidate record containing original answer text, candidate ID, model identity, completion status, evidence/citation references, usage, timing, and raw-response reference where retention is permitted.

## Simple example

Provider A places text in one field; Provider B returns several content blocks. Both become an answer-text field while preserving their original structure for debugging. A refusal remains a refusal.

Unknown token usage stays unknown. Zero means a measured zero; it should not stand in for missing information.

## Connection to the papers

A shared record makes the papers' comparison ideas reproducible. This exact record design is our engineering proposal, not a method claimed from one of the supplied papers.

