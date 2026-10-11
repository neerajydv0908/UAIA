# 5. Retrieve authoritative evidence

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It finds passages that can answer the question and preserves enough information to check them later. **RAG** means retrieving information and giving it to a generator when it writes.

**Input:** question, source/freshness needs, permitted collections, and retrieval budget.

**Output:** an evidence package with passage IDs, exact text, source URL or document/page, publisher, dates when known, retrieval time, and version/hash. A hash is a fingerprint used to notice content changes.

## Simple example

For Device Z's operating range, retrieve the operating section, not merely a similar storage-temperature passage. Keep surrounding conditions so “up to 40°C when dry” does not become an unconditional claim.

Fresh or document-specific evidence normally comes before step 4. Step 8 may request a bounded targeted follow-up if an important claim remains unresolved.

## Connection to the papers

PDE-Agents supports testing early evidence provision and input fidelity. Its graph-based experiment does not prove we need a knowledge graph. Compare simple passages before adding one. [Paper review](../../../research_papers.md).

