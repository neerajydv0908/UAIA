# 3. Select suitable answer models — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | How it chooses | Strength | Trade-off |
| --- | --- | --- | --- |
| One fixed model | Always call the same model | Essential low-complexity baseline | No alternative answer if it fails |
| Fixed pair in parallel | Always use the same two generators | Clean comparison data; often less waiting than a cascade | Pays for both even on easy questions |
| Task-based rules | Map task categories to tested models | Clear and inexpensive decisions | Stale rules can miss provider changes |
| Cheap-first cascade | Ask one model; escalate when defined checks require it | May avoid unnecessary calls | Escalations add sequential waiting; the second model can also be wrong |
| Learned router | Predict quality/cost from past outcomes | Can adapt to observed task patterns | Needs representative labels for alternatives |
| Contextual bandit | Learn choices while occasionally exploring alternatives | Research option for changing traffic | Complex evaluation and user-impact controls; defer |

A cascade is a sequence of attempts. A contextual bandit balances using a currently preferred option and trying another to learn about it.

If logs only contain outcomes from the model we selected, we do not know what the uncalled models would have done. Collect controlled comparison data before claiming learned routing improvements.

Require capability compatibility before optimizing cost. A low-cost model unable to accept the necessary input is not a suitable candidate.

