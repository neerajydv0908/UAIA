# 9. Judge and score the answers — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative judge strategies

| Strategy | How it works | Benefit | Limitation / cost |
| --- | --- | --- | --- |
| Rules and verifier outcomes | Apply explicit requirements and completeness checks | Transparent baseline | Limited understanding of nuanced answers |
| Prompted pairwise judge | Compare A and B directly using evidence | Natural choice for two candidates | Order, style, and self-preference bias; extra call |
| Dimension scoring | Score support, completeness, relevance, and presentation separately | Shows why an answer wins or fails | Numeric scales may vary between calls |
| Several independent judges | Compare judgments, escalate disagreement | Can reveal unstable decisions | More cost; errors may be correlated |
| Existing open reward/judge model | Run a pretrained scoring model locally | Useful fixed baseline | Its training may not match our sources or task |
| Our evidence-conditioned judge | Train on our question/evidence/answer labels | Can target our product's failure patterns | Needs good data and independent evaluation |

For n candidates, comparing every pair requires n(n−1)/2 comparisons before repeats. A tournament reduces calls but can be sensitive to ordering. Begin with two candidates.

## What “our own judge” can mean

1. **Small scorer from random weights:** learns from features such as central-claim support and failed checks. Cheap learning experiment; cannot independently read arbitrary text.
2. **Adapted pretrained text judge:** reads question, evidence, and answer. Train support classification and/or a reward score using reviewed data.
3. **Scratch-trained language model:** first learns language from text, then learns judging. Valuable research/learning path; far larger burden than training a small scorer.

These are different experiments. Adding evidence to a generic reward model's prompt is only a baseline, not proof of evidence-conditioned competence.

## Proposed training objective

A pairwise reward model learns to give the preferred answer a higher score for the same question and evidence. The score difference expresses relative preference; it is not automatically the probability that an answer is true. Preserve ties and neither-adequate cases with an explicit labeling/training policy rather than inventing winners.

An additional support classifier could learn claim-level supported/contradicted/unknown labels. Test it separately before combining its output with ranking.

AdaRubric's DPO improves a generator from preferences. DPO is not a required recipe for training our judge.

