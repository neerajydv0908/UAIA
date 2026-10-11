# 10. Select an answer or synthesize one

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It decides what text should become the proposed final answer. Selecting keeps an existing answer; synthesizing writes a new one from supported material.

**Input:** eligible answers, judge outcomes, verified claim/evidence records, and the task's completeness requirements.

**Output:** a selected or newly composed draft, its provenance, and any unresolved gaps. Provenance means knowing where each statement came from.

## Simple example

Answer A gives the supported operating range; answer B includes a supported humidity restriction. Combining them may help—but only if both refer to the same device version and conditions.

If every candidate gives an unsupported range, voting or merging cannot manufacture a supported answer. Additional retrieval or a limited uncertainty response may be needed.

## Connection to the papers

Judge Anything reminds us that choosing between answers differs from absolute quality scoring. The papers do not establish that synthesis always beats selection. We propose testing synthesis as a separate change.

