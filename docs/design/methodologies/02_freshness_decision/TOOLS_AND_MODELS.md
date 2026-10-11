# 2. Decide whether fresh information is needed — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Option | Role |
| --- | --- |
| Python decision table and date handling | Store rules, requested dates, and reasons |
| Shared task classifier from step 1 | Add a freshness suggestion in the same call when appropriate |
| SQLite records | Track source versions, retrieval times, and cache decisions |
| Search-service adapters from [step 5](../05_evidence_retrieval/TOOLS_AND_MODELS.md) | Execute the chosen search later |
| A learned classifier | Future option using reviewed “search needed?” labels |

We do not need a special freshness LLM or a separate model call for every request. Sharing classification can save calls, provided each field is still evaluated.

The [MCP architecture](https://modelcontextprotocol.io/docs/learn/architecture) supplies a tool interface. The decision to call a search tool remains our application's responsibility.

