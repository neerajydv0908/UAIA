# 8. Verify claims, citations, calculations, code, and safety — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternatives by checking job

| Job | Simpler method | More advanced method | What neither alone establishes |
| --- | --- | --- | --- |
| Citation identity | Confirm passage ID and source link | Check version, original publisher, and provenance | That the text supports the claim |
| Factual support | Human review of passage and claim | NLI model or evidence-aware LLM, with targeted retrieval | Truth beyond the available sources |
| Calculations | Restricted arithmetic and unit checks | Symbolic equivalence, constraints, numerical cross-checks | Correct problem interpretation |
| Code | Syntax/static checks | Independent tests in an isolated environment | All possible program behavior |
| Safety | Explicit product/tool rules and reviewed cases | Moderation classifier plus escalation | Zero harmful misses or zero false alarms |
| Source quality | Domain-specific source policy | Conflicting-source review and provenance analysis | Infallibility of an official source |

NLI, or natural language inference, compares a passage with a statement and predicts support, contradiction, or uncertainty. It is evidence interpretation, not web search.

## Combining checks sensibly

Use exact checks when the question permits them. Use source interpretation for factual prose and retain uncertainty. Ask a second judge only for meaningful unresolved cases, within budget.

A failed central requirement can make an answer ineligible. An unknown check must not become a pass. A missing fact is not automatically false. A safety flag is separate from factual correctness.

## Execution boundaries

Generated code must not run on the user's host with project secrets. Use an isolated, disposable environment with CPU/time/memory limits and appropriate network/file restrictions. Tests should include independent cases; generator-written tests alone can share the generator's mistakes.

Do not evaluate arbitrary model text as Python or unrestricted symbolic expressions. Parse a restricted calculation format first. For web fetching, restrict destinations and resource use so an answer cannot direct tools into private services.

