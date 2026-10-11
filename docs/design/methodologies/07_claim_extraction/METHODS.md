# 7. Extract checkable claims — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | How it works | Strength | Limitation / cost |
| --- | --- | --- | --- |
| Manual annotation | A person marks claims and qualifiers | Best initial teaching/reference set | Time-consuming |
| Sentence/rule splitting | Separate sentences and recognize numbers/links | Cheap, easy baseline | One sentence may contain multiple claims; pronouns depend on context |
| Structured LLM extraction | Request a list of atomic claims and original spans | Flexible for natural language | May omit, alter, or invent a claim |
| Two-pass extraction and audit | Extract, then check coverage and faithfulness | Can expose missed qualifiers | More calls; two models can share errors |
| Trained extractor | Learn from labeled spans and claim structures | Potential repeated-use efficiency | Requires carefully labeled domain data |

“Atomic” means small enough to assess as one statement. It does not mean stripping away conditions until the sentence changes meaning.

## What must survive extraction

Retain negatives, exceptions, dates, comparison baselines, units, and scope. “Some tests passed” must not become “all tests passed.” Resolve a pronoun only when its referent is clear.

Keep unsupported citation placeholders attached to the claim so later checks can detect them. Do not allow a candidate answer's instructions to control the extractor.

Prioritize central claims when resources are limited, but report the unchecked remainder. A short extracted list is not proof that the answer contains only those claims.

