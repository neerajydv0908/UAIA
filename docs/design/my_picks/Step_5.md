# Step 5: Retrieve evidence

Chosen for the first prototype: 2026-09-28. These are our current engineering preferences, not measured universal winners.

[All picks](README.md) · [Alternative methods](../methodologies/05_evidence_retrieval/METHODS.md)

## My preferred method

A tiny saved JSON passage collection with keyword overlap; choose direct retrieval or a real read-only MCP connection.

## Model choice

None initially. Later retrieval comparisons: BGE-M3 or text-embedding-3-small, then a reranker if needed. Perplexity Search is the preferred initial web-service trial, pending budget and source-handling work.

## Why prefer this now?

The same small corpus works through either interface, making MCP's role easy to see. We learn source tracing before adding web indexing and embeddings.

**Why the alternatives wait:** SQLite/BM25 is the next scale-up from the in-memory teaching retriever. Dense, hybrid, and graph retrieval remain alternatives, not required infrastructure.

## What is actually implemented?

Local retrieval and the official Python MCP SDK client/server are implemented and integration-tested. Web search/fetching is pending.

[Read the relevant code](../../../answerlab/retrieval.py). The [build status](../../build_status.md) distinguishes working code, mocked API tests, and future work.

## Research connection

PDE-Agents motivates evidence before generation and testing input fidelity; it does not require a graph database. See [the paper explanations](../../research_papers.md).

## Evidence and next decision

Direct and MCP retrieval return the same passages. Duplicate source IDs and absent evidence are handled.

Keep this choice until a recorded comparison shows a useful improvement in quality, cost, or time. Use grouped development/test examples and the [evaluation protocol](../../evaluation.md). Model IDs and API capabilities must be checked again before live use; the [catalog](../methodologies/MODEL_CATALOG.md) contains official sources.

