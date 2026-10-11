# 9. Judge and score the answers — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Option | Use and caveat |
| --- | --- |
| Python eligibility rules | Apply central correctness requirements before style |
| Hosted model from the [catalog](../MODEL_CATALOG.md) | Prompted pairwise/rubric judge; benchmark separately from generation |
| Skywork-Reward-V2-Qwen3-4B | Existing scalar-reward baseline; test evidence sensitivity |
| Prometheus 2 | Rubric-grading baseline with the documented reference-answer requirement |
| scikit-learn or a small PyTorch network | Our initial feature-based scorer |
| Pretrained Qwen + Transformers/TRL | Candidate route to a learned text reward model |
| Human annotation | Independent reference judgments; do not use our own judge as its only examiner |

Primary references: [Skywork card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B), [Prometheus card](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0), [TRL reward modeling](https://huggingface.co/docs/trl/reward_trainer), [PyTorch learning guide](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).

Data starting points: HelpSteer2-Preference for general preferences, RAGTruth for grounding errors, and our reviewed evidence-linked pairs for the actual product. Public evaluation sets remain separate. See [DATASETS.md](../../../datasets.md).

