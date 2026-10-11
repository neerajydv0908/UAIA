# How to mark a source-based answer pair

Read the question and source before deciding which answer is better. Suggested labels are for discussion, not a completed independent review.

## Five possible decisions for our evidence task

| Decision | Meaning |
| --- | --- |
| A | A is adequate and better supported for the question than B |
| B | B is adequate and better supported for the question than A |
| `tie_adequate` | Both adequately answer the question using the available evidence |
| `neither_adequate` | Both have identifiable defects that make them inadequate, such as contradicting supplied facts |
| `insufficient_evidence` | Available material does not let us make the necessary factual decision |

If both answers are defective, do not reward one just because it is slightly less defective. If the source is missing a fact, do not invent it. If an answer correctly explains a lack of evidence, that can be an adequate response; our examples with two unsupported guesses are a different case.

For each answer, separately note support, contradiction, missing support, relevance, and whether it answers all important parts. Preference alone is not enough to describe those properties. Later versions should store these dimensions explicitly before computing any combined score.

## A review, step by step

1. Read what the user actually asked.
2. Read all supplied source text, including dates, exceptions, units, and conditions.
3. Check A's important statements against it. Repeat for B.
4. Check whether each answer actually addresses the question.
5. Choose one of the five decisions and write a short reason tied to a source passage.
6. Record the reviewer, date, source IDs, and any disagreement. Keep the original suggestion separately.

Do not mark a model-suggested judgment as human-reviewed without a real person checking it. Obtain a second independent review on an initial subset. Resolve disagreements or retain uncertainty. Hide provider identities and the other reviewer's judgment during independent review. To avoid answer-position habits, vary which answer is A; when swapping, swap the corresponding label as well.

The current 20 practice examples are balanced across the five outcomes to make the rubric easy to follow. That artificial balance is not an estimate of real user traffic. Expand with real, permitted questions and naturally occurring mistakes after the labeling rules make sense.

## Before an actual training run

Check a sample of public labels, review the custom examples, define the first user task, and finish the similar-question/shared-source overlap audit. Keep all versions of a question or source together when assigning portions. Check uses and retention permissions for any new material. Remove private details that are not needed.

The first 100 public rows are for training-side inspection; never append that file to the training file or you will count its rows twice. Leave calibration and test aside while choosing examples or prompts. No training or automatic feedback learning is part of this milestone.
