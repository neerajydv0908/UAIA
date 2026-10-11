# MY Pick — our current choices

Updated: 2026-09-28. This folder records the approach we prefer for each step, why alternatives wait, and what is implemented.

The first build is a **local document-quotation prototype**. It is an executable learning foundation for the full product, not a trained judge or a production answer-verification service. A narrow rule baseline makes later improvements measurable.

| Step | Current preference | Status |
| --- | --- | --- |
| [1. Understand the question](Step_1.md) | Readable rules with multiple task labels and an explicit clarification flag. | See step file for implemented scope and pending upgrades |
| [2. Decide which information is needed](Step_2.md) | Explicit source rules: saved documents, fresh web, no retrieval, or uncertain. | See step file for implemented scope and pending upgrades |
| [3. Select candidate models](Step_3.md) | A configurable fixed provider pair with call limits and a budget check before network work. | See step file for implemented scope and pending upgrades |
| [4. Call providers concurrently](Step_4.md) | Python asyncio and HTTPX, independent provider outcomes, bounded calls/time, and no automatic retries. | See step file for implemented scope and pending upgrades |
| [5. Retrieve evidence](Step_5.md) | A tiny saved JSON passage collection with keyword overlap; choose direct retrieval or a real read-only MCP connection. | See step file for implemented scope and pending upgrades |
| [6. Normalize provider responses](Step_6.md) | Explicit provider adapters and Pydantic records, preserving original text, returned model ID, finish status, usage, and timing. | See step file for implemented scope and pending upgrades |
| [7. Extract claims](Step_7.md) | An explicit quotation format: one complete source passage and its citation per answer line. Every nonempty line becomes a check record. | See step file for implemented scope and pending upgrades |
| [8. Verify evidence and applicable outcomes](Step_8.md) | Exact complete-passage matching against the cited saved source. Missing citations and changed wording remain unknown. | See step file for implemented scope and pending upgrades |
| [9. Judge answers](Step_9.md) | Required quotation checks before ranking; simple relevance overlap with a visible tie outcome. | See step file for implemented scope and pending upgrades |
| [10. Select or synthesize](Step_10.md) | Select an eligible original answer; break equal heuristic scores by concision and a stable tie-break while preserving the tie record. | See step file for implemented scope and pending upgrades |
| [11. Recheck the final text](Step_11.md) | Re-extract and check the exact outgoing text, then compare claim/source fingerprints with its prior checks. | See step file for implemented scope and pending upgrades |
| [12. Present the result and uncertainty](Step_12.md) | A fixed CLI presentation with source references, an optional 13-step trace, and explicit uncertainty reasons. | See step file for implemented scope and pending upgrades |
| [13. Learn from feedback](Step_13.md) | Explicit local feedback storage with separate storage/training consent, pending human review, and no automatic model updates. | See step file for implemented scope and pending upgrades |

Use [BUILD_STATUS.md](../../build_status.md) for honest implementation coverage. The larger [methodology library](../methodologies/README.md) remains our alternatives reference.

The live API pair is a preferred trial, not a measured winner. No real provider calls, web searches, or training jobs have been performed by this build. API prices are intentionally blank and live spending defaults to zero.

When a choice changes, update its step file.

