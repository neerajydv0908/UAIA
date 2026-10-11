# 4. Ask models concurrently

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It sends requests and collects their outcomes reliably. **Concurrent** means several requests can be in progress at once; it does not mean our computer trains several models.

**Input:** selected providers, common question/evidence package, allowed generation settings, and reserved budget.

**Output:** one result per attempted provider: completed, timed out, refused, truncated, failed, or cancelled, plus recorded usage and timing when available.

## Simple example

Model A finishes while Model B times out. Keep A's answer and record B's timeout. The judge receives the actual number of usable answers.

Give generators the same relevant evidence when comparing their ability to answer from it. If a provider uses its own search, record that as a different condition.

## Connection to the papers

The activation paper motivates measuring cost against quality. PDE-Agents motivates separated responsibilities, but its supplied implementation is sequential; it is not evidence that our concurrent design is already validated.

