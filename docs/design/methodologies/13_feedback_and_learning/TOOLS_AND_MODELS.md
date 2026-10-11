# 13. Learn from reviewed feedback — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Tools and model options

| Tool | Role |
| --- | --- |
| JSONL / SQLite and versioned manifests | Small, inspectable feedback and dataset records |
| Label Studio | Optional human annotation interface |
| scikit-learn | Small feature scorer and router baselines |
| PyTorch | Learn and implement training from random weights |
| Transformers + TRL RewardTrainer | Candidate practical reward-model training route |
| PEFT/LoRA integration | Adapt fewer parameters of a pretrained model |
| Saved experiment reports | Track dataset, prompt/model revision, metrics, cost, and decision |

Sources: [Label Studio](https://labelstud.io/guide/), [PyTorch beginner guide](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [TRL reward modeling and adapter support](https://huggingface.co/docs/trl/reward_trainer).

## Dataset menu by purpose

| Purpose | Starting option | What not to assume |
| --- | --- | --- |
| General answer preference | HelpSteer2-Preference; optional UltraFeedback | Preference is not proof of factual support |
| Source grounding | RAGTruth; targeted FEVER/SciFact exercises | Span/claim labels are not whole-answer preference pairs |
| Our actual product | Reviewed user-domain evidence-linked examples | A thumbs-up is not a trusted label |
| Scratch language learning | TinyStories or another permitted small corpus | Stories do not teach reliable evidence verification |
| Independent evaluation | RewardBench 2; task-specific EvalPlus/GSM8K where relevant | Evaluation data is not free training data |

Dataset links, license/split details, contamination risks, and proposed record fields are maintained in [DATASETS.md](../../../datasets.md). Exact models are in the [catalog](../MODEL_CATALOG.md). No dataset has been downloaded or approved for training here.

