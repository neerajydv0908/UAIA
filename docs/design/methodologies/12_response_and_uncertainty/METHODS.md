# 12. Return the answer, citations, and uncertainty — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative presentation strategies

| Method | What the user sees | Strength | Limitation |
| --- | --- | --- | --- |
| Fixed template | Answer, citations, key limitation, optional check summary | Predictable and easy to audit | Less flexible for varied tasks |
| Model-written presentation | A model turns records into prose | Flexible wording | Can add claims; must return to step 11 |
| Per-claim source view | Click a statement to inspect its evidence and status | Good traceability | More interface work; can overwhelm |
| Qualitative uncertainty | Reasons such as missing, conflicting, or dated evidence | Useful before calibration | Labels require consistent definitions |
| Calibrated probabilities | An estimated chance of a specifically defined outcome | Can support measured thresholds | Needs sufficient independent labels and ongoing validation |

**Calibration** means checking whether predicted probabilities match observed outcomes. If a system assigns 80% to a well-defined event across many similar cases, approximately 80% should satisfy that event.

Do not label a reward score, average judge confidence, citation count, or provider agreement as a correctness percentage.

## Define the event before estimating its probability

Possible future event: “all central factual claims satisfy our independent review rubric.” That differs from “this answer is preferred to candidate B” and from “this single passage supports this single claim.”

Measure uncertainty by task and answer coverage. A system that refuses nearly everything can have few errors among answered requests while being unhelpful.

Citations should appear next to the claims they support. A single source list at the end can obscure which claim was checked.

