# 11. Reverify the outgoing answer — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Tool | Job |
| --- | --- |
| Step 7 extractor and step 8 verifiers | Reassess the final content using the same explicit statuses |
| Text/evidence hashes and version IDs | Identify whether a cached check still applies |
| Python comparison and citation-mapping checks | Detect altered text and broken references |
| A separately evaluated prompted critic | Optional additional review on unresolved cases |
| Simple state machine; LangGraph later if warranted | Track draft → checked → repaired → checked → released/abstained |

A state machine is a list of allowed states and transitions, preventing paths such as “repair → release without checking.”

No new general LLM is required just for this step. Sources for applicable tools are in [verification](../08_verification/TOOLS_AND_MODELS.md) and [workflow options](../04_concurrent_requests/TOOLS_AND_MODELS.md).

