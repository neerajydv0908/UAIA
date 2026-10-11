# Why we selected this model

**Qwen/Qwen3-4B-Instruct-2507** was chosen on 2026-10-05. It will be the starting model for our judge experiment.

## What the name means

`Qwen3` identifies the model family. `4B` means roughly four billion learned numbers. `Instruct` means it has already been taught to follow instructions. This particular release uses the non-thinking output mode. The publisher lists Apache 2.0 as its license. [Official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Those are model properties, not evidence that it is already a reliable judge. We will measure its judging performance ourselves.

## Our exact choice

| Item | Selection |
| --- | --- |
| Publisher model | `Qwen/Qwen3-4B-Instruct-2507` |
| Fixed revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Files | Original BF16 safetensors; no community conversion for this first run |
| First runtime | MLX-LM on this Apple silicon Mac |
| First output | A, B, or `tie_preference` |
| Training starting collection | HelpSteer2 v2: 6,623 training pairs |
| Planned teaching method | LoRA: adjust a small added part of the model |
| Scratch training | Optional separate learning exercise |

A **revision** is a fixed saved version. Keeping that long identifier means we can repeat the same experiment even if the publisher updates its page. **Weights** are the learned numbers stored in the large files. **BF16** is how these original files store those numbers; it is not a quality score. The saved model is about 8 GB, but running or training also needs working memory.

MLX-LM supplies Apple silicon model-running tools. We pin the installed package and record its dependency versions. [Official MLX-LM documentation](https://github.com/ml-explore/mlx-lm).

## Why this route fits our next experiment

The model was already chosen, and it gives us a concrete starting point. We avoid spending weeks comparing names before running one small test. Keeping the original files gives us a clear starting version. If measured training memory is too large, we can test a compressed copy of the same model and measure whether that changes its decisions.

LoRA keeps most existing numbers fixed and learns a small added set. A compressed-model version of that method is called QLoRA. Neither method guarantees improved judging; both need an exam against the starting model. [Official training guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

## What our first judge will learn

The first task is general answer preference: read the question and two answers, then choose A, B, or a tie. The public data supplies human preferences, not verified source-grounding labels. Later, we will teach source-based judgments using permitted evidence and reviewed decisions.

The local exercises include source text to make them easy to check. They are authored teaching examples, not training data or a product benchmark. We must still measure ordinary preference behavior before and after training.

## Keep these limits clear

The publisher's maximum input length is not our memory budget. The first teaching script refuses inputs over 2,048 text pieces and limits the answer to 16 pieces. We have not established training memory or speed yet. A **token** is one of those small text pieces.

No numeric confidence or factual-certification claim follows from an A/B choice. A/B reversal and equal-answer examples are initial checks, not a comprehensive bias test. Permission notices and source attribution are kept with the downloaded files; later sharing of a trained model must preserve applicable notices.
