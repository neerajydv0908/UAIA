# Dataset sources and our current choices

Checked 2026-09-30. A published license label is recorded here, not a blanket clearance of every underlying item or eventual product use. No paid generation or model training occurred.

| Source | What it contributes | Decision |
| --- | --- | --- |
| [NVIDIA HelpSteer2-Preference](https://huggingface.co/datasets/nvidia/HelpSteer2#preferences-new---1-oct-2024) | Two answers and human preferences, with reasons | Downloaded first: smaller, straightforward pair format. General preference task only. Card declares CC BY 4.0. |
| [NVIDIA HelpSteer3-Preference](https://huggingface.co/datasets/nvidia/HelpSteer3#preference) | A broader collection of human comparisons, including several domains and languages | Later candidate, not downloaded. Card describes 40,476 preference samples and CC BY 4.0. Newer does not automatically mean better for our specific judge. |
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | Source material and human-marked unsupported/contradictory answer spans | Useful research reference for grounding. Do not put its QA portion into our commercial training pool without resolving underlying-source permissions. |
| Our own evidence comparisons | Source material we can use, answers, and reviewed decisions | Central collection to build. Current 20 examples are fictional model-written drafts, not production-quality human labels. |
| [RewardBench 2](https://huggingface.co/datasets/allenai/reward-bench-2) | An external test of judge behavior | Evaluation candidate only, not training data. Card notes ODC-BY, research/educational intent, and separate third-party output terms. |

## Why RAGTruth needs care for this product

RAGTruth's repository lists MIT, but its QA records use MARCO source material. [MS MARCO's published terms](https://microsoft.github.io/msmarco/) specify non-commercial research and state that underlying document rights may not be owned by the distributor. Repository licensing alone does not resolve that source-content question. The other RAGTruth tasks use different sources and need their own review; this is not a claim that every RAGTruth item has identical terms.

## Exact downloaded snapshot

- Publisher: NVIDIA; dataset: HelpSteer2; configuration: preference.
- Revision: `990b2711a36180dd19d9c94b8627844866f8982a`.
- [Original compressed preference data](https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/990b2711a36180dd19d9c94b8627844866f8982a/preference/preference.jsonl.gz).
- [Original dataset card](https://huggingface.co/datasets/nvidia/HelpSteer2/blob/990b2711a36180dd19d9c94b8627844866f8982a/README.md).
- Download size: 15,339,272 bytes for the compressed data, about 15 MB.
- [Local download manifest](raw/helpsteer2/download_manifest.json): URLs, retrieval time, version, sizes, and file fingerprints.
- [Saved source card](raw/helpsteer2/DATASET_CARD.md): retained publisher documentation and citation information.

Our changes: field renaming, preserving human labels and reasons, explicit missing-evidence fields, exclusions, and project splits. There is no rewriting of the original questions or answers. Retain NVIDIA attribution, the source/version, [the CC BY 4.0 license link](https://creativecommons.org/licenses/by/4.0/), and this change notice when sharing an authorized derivative. The source card contains the authors' preferred paper citations.

Do not combine HelpSteer2 and HelpSteer3 without checking overlap. A preference tie means similar preference; it does not establish that both answers are correct. The first collection is manageable on this computer without installing a training framework.

## Additional changes in v2 — 2026-10-05

A bounded text-overlap audit and model-assisted sample review removed 558 additional rows from the experimental release, leaving 8,563. Original questions, answers, upstream labels, and the portion assignment of retained rows remain unchanged. Quarantine IDs and reasons are retained. The publisher attribution, source revision, and license record above continue to apply.
