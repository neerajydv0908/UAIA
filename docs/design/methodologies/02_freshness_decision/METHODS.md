# 2. Decide whether fresh information is needed — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | Description | Advantage | Limitation / cost |
| --- | --- | --- | --- |
| Always search | Retrieve for every request | Very simple routing baseline | Wasteful for creative tasks or arithmetic; irrelevant search can distract |
| Never search | Use supplied context and model memory | Cheap comparison baseline | Cannot reliably support current facts |
| Explicit freshness rules | Search for changing topics, explicit dates, or requested verification | Understandable and cheap | Needs maintenance and careful handling of context |
| Prompted decision | Ask a model whether outside/current information is needed | Flexible wording | The model may be wrong about what it knows |
| Hybrid with source-age policy | Rules/classifier decide need; source-specific age limits decide reuse | Balances freshness and reuse | Must handle stale, incomplete, or misdated records |

A cache is saved material reused to avoid another request. TTL means how long that saved material is considered reusable. We should not set one universal TTL for weather, manuals, and historical facts.

## Important distinctions

A page's fetch time is not its publication date. A page fetched today can contain an old price. Conversely, an old official definition can still be suitable for a stable question.

Source relevance, authority, and freshness are separate checks. A recent low-quality page should not automatically outrank an authoritative version that answers a historical question.

