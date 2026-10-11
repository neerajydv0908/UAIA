# 11. Reverify the outgoing answer — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | How it works | Benefit | Limitation / cost |
| --- | --- | --- | --- |
| Full re-extraction and recheck | Run claim extraction and applicable checks on all final text | Clear initial logic | Repeated model/tool work |
| Reuse unchanged checks | Reuse results whose text, evidence, settings, and versions match | Avoids needless repeated work | Requires strict identity and freshness rules |
| Changed-claim checks plus regression checks | Check edits deeply and scan the whole answer for lost scope | May reduce cost | A small edit can change another sentence's meaning |
| Independent final critic | Another model reviews the draft and evidence | Can catch different mistakes | May share bias; a critic is not proof |
| Stronger formal/task checks | Validate a derivation, constrained formula, or program property | Strong evidence within a narrow scope | Cannot generally certify arbitrary prose |

A regression is a mistake introduced after an earlier version worked.

## Bounded repair

Proposed rule: at most one targeted repair attempt in the early prototype, followed by checking the repaired answer. This is a configurable starting limit, not a proven optimum.

Do not loop until a judge happens to say “pass.” If a required check still fails, use a previously verified eligible answer if it remains valid, otherwise return a narrower response or abstention.

Any subsequent model-written wording change returns here. Formatting that changes only presentation can be handled mechanically, but changes to numbers, claims, or citation attachment invalidate relevant checks.

