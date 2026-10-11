# 4. Ask models concurrently — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | How it works | Advantage | Limitation |
| --- | --- | --- | --- |
| Sequential calls | Ask A, then B | Easiest to debug | Waits accumulate; useful baseline, not concurrent execution |
| Bounded asynchronous calls | Start selected requests together, limiting how many can run | Small, direct solution for network waiting | We must handle timeouts, budgets, cancellation, and failures |
| Common provider gateway | Call providers through one interface | Less repeated adapter plumbing | Added dependency; provider differences can be hidden |
| Workflow graph | Define named steps, branches, and stored state | Useful when repair/escalation paths become complex | More concepts and setup |
| Durable background jobs | Store jobs so workers can resume after interruption | Useful for longer product requests | Requires persistence, duplicate handling, and operational work |

A semaphore is a limit on simultaneous work. Backoff means waiting before retrying a temporary failure. A durable job survives a process restart.

## Reliability choices

Set both per-call timeouts and a whole-request deadline. Reserve spending before launching parallel work to avoid each call independently consuming the full allowance. Count all attempts, including uncertain or failed ones.

Retry only suitable temporary errors and cap attempts. Blindly retrying a request with an uncertain billing outcome can double cost. Do not assume cancelling our wait cancels the provider's computation or bill.

One provider failure should become its own result instead of erasing successful siblings. Keep concurrency limits per provider as well as for the whole service.

