# What one saved example means

Think of each example as a worksheet with a question, reading material, two answers, and a marking section. Keep the marking section hidden from the model while it answers.

| Field | Plain meaning |
| --- | --- |
| `id` | A name for this example |
| `group_id` | A shared name for related questions so they stay in the same portion |
| `split` | Its purpose: practice, settings checks, confidence checks, or the held-out exam |
| `task` | Which skill the label actually teaches |
| `input.question` | The question the model reads |
| `input.evidence` | Supplied source passages; an empty list means none was separately supplied |
| `input.answers.A` / `B` | Original answers to compare |
| `target` | Publisher-derived preference label; supervise with it, never include it in input |
| `grounding_label` | Whether source support was annotated; public preference rows say `not_annotated` |
| `review_status` | Who has reviewed this example so far |
| `provenance` | Where it came from, its version, original line, and declared license |
| `audit_only` | Original preference scores, reasons, and annotator details for inspection, not inference input |

Only the `input` fields are intended as model inputs in a later training implementation. Showing it the winner, reasoning, or metadata that reveals the label would be like giving a student the answer key during an exam. Even inputs must later be checked for accidental label clues and embedded source text shared across portions.

The HelpSteer2 original preference strength runs from -3 to 3: negative favors response 1 (our A), positive favors response 2 (our B), and zero means a preference tie. We preserve the original strength and all released annotator details. We do not turn this scale into a truth probability.

The fictional practice examples have a different shape on purpose: `proposed_target` is a model suggestion; `reviewed_target` is null, meaning a reviewer has not supplied a final label. `training_eligible` is false. Their groups are unassigned until review and collection design; they are not a held-out test.

The product application does not train during user requests. A separate [training preparation and trial](<../model selection/TRAINING_TRIAL.md>) now converts training-only records into teaching conversations and uses their labels as targets. This folder preserves the normalized source records.
