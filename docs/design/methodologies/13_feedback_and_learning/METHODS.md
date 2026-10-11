# 13. Learn from reviewed feedback — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative learning strategies

| Strategy | What changes | Data needed | Main trade-off |
| --- | --- | --- | --- |
| Reviewed rule/prompt updates | Improve rubric wording, tools, or retrieval settings | Categorized failures | Often useful before any training; still needs regression evaluation |
| Feature scorer from random weights | Learn a small ranking function from checker features | Reviewed outcomes and reliable features | Low-cost learning; cannot independently understand raw text |
| Fine-tuned pretrained judge | Teach a text model to assess evidence and rank answers | Question, evidence, candidates, preferences/support labels | Practical custom-judge route; retains pretrained biases |
| Full fine-tuning vs LoRA | Update all pretrained weights vs a smaller added set | Same well-defined labels | Different memory/compute demands; neither is scratch pretraining |
| Scratch language-model training | Learn language from random weights, then learn judging | Broad permitted text followed by judging data | Educational research path with much larger data/compute burden |
| Learned router | Improve which models/tools get called | Comparable logged outcomes and costs | Uncalled alternatives have missing outcomes |
| Online exploration | Try alternatives on controlled traffic | Logging, safeguards, and enough traffic | Later research; user votes alone are too noisy |

LoRA adapts a pretrained model by training a smaller added set of parameters. It can reduce training resources, but the resulting model still depends on its pretrained base.

## Three learning tracks

**A. Our first scratch-trained scorer:** start with a small number of features, predict preference/adequacy, measure errors, adjust weights, and inspect held-out results. This teaches ML without pretending to be a language model.

**B. Practical evidence-conditioned text judge:** adapt a pretrained model. Keep evidence in the input for both candidates; supervise support and/or comparison. Compare full fine-tuning with LoRA only after a tiny working experiment.

**C. Educational scratch LLM:** tokenize permitted text, train a small next-token predictor from random weights, evaluate its language learning, then teach judging. TinyStories is a possible learning corpus; preference pairs alone do not provide general language pretraining.

## Reviewed improvement cycle

Collect permitted records → remove secrets and duplicates → review labels → assign grouped splits → train or revise → evaluate independently → release a version if it earns adoption → monitor and retain rollback.

A rollback restores an earlier working version. Never silently train on the final test set or repeatedly tune until its score improves.

Sample both difficult/disagreed cases and some ordinary requests. Feedback only from unhappy users or only from model disagreements creates a distorted picture.

