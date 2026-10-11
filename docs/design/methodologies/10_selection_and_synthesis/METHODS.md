# 10. Select an answer or synthesize one — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Strategy | What happens | Benefit | Risk / cost |
| --- | --- | --- | --- |
| Select original | Return the best adequate candidate after final checks | Simple; preserves tested wording | May omit useful material in another answer |
| Extractive assembly | Combine exact supported sentences or spans | Easier provenance | Awkward flow; juxtaposition can still imply a false connection |
| Evidence-grounded synthesis | Ask a model to compose using allowed claims and sources | Can improve completeness and readability | New unsupported claims, lost caveats, extra call |
| Critique and revision | Let a critic identify gaps and a writer revise | May fix concrete weaknesses | More iterations; persuasive criticism may be wrong |
| Debate/consensus | Models respond to each other | Research comparison for disagreement | Agreement can increase without truth; expensive |
| Abstain or answer partially | State only what is supported and explain the missing part | Honest handling of all-bad candidates | Less complete; must still be useful |

“Best among candidates” and “adequate to return” are different tests. Compare eligible candidates only after checking essential requirements.

## For any synthesis method

Pass the question, approved evidence, supported claim IDs, relevant conditions, and unresolved conflicts. Candidate text is a proposal, not new authority.

Do not resolve a contradiction by averaging numbers from incompatible dates or definitions. Avoid merging policy, operating, or pricing conditions that belong to different versions.

Keep a mapping from final claims to source passages. Step 11 checks the resulting text, including connective sentences that may introduce new implications.

