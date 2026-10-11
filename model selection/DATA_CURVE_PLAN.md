# Next judge experiment: measure the data curve

**Superseded on October 7:** use [one full-data A100 run](A100_FULL_DATA.md). The three-size jobs and pending eight-hour choice below are preserved as proposal history, not the current execution plan. The frozen 400-pair monitor and prepared data remain in use.

October 7, 2026. **Prepared locally, not run on a GPU.** The previous corrected judge remains **252/311 (81.0%)**, with log-loss 0.450. No new accuracy is claimed. This preparation replaces the single proposed 2,000-pair run with the user's staged data-size comparison.

## What is ready

1. **A separate 400-pair monitor.** Keep all 200 old held-out pairs, then select 200 additional training-side groups outside the original 2,000-pair pool. Fill deficits in preference-strength, shorter-winner and multi-turn strata using training metadata only. Exclude every selected group from every training mode. Freeze this monitor for all following comparisons.
2. **Reproducible shuffling.** Keep the recorded first-pass order; reshuffle later passes using a separate seeded random generator. Both answer orders of each pair stay in the same update. Resume reconstructs the identical remaining order, without consuming dropout's random state.
3. **Optional strength weights.** `--weight-strength` multiplies each clear pair's loss weight by 1, 2 or 3, then normalizes the pool mean to one. This retains the strength ratios without tripling the average loss scale. It is off for the data curve and rejects aggregate ties. This is a controlled hypothesis inspired by scaled Bradley–Terry, not a claim of reproducing NVIDIA's model or result. [Original paper](https://arxiv.org/html/2410.01257v1).
4. **Explicit data sizes and seeds.** `--max-pairs` selects a recorded prefix of the eligible pool; `--seed` controls initialization and later shuffling. The plan saves every training pair ID and batch order. Nested 1,200/2,400/full subsets therefore share the same initial order.
5. **A minimum complete pass.** `--minimum-epochs 1` prevents patience from stopping an intended full-data comparison after only a small prefix. `--patience 2` remains available afterward; an external time limit always takes precedence. This guard was tested alongside interrupted/resumed training across a shuffled epoch.

The versioned export now contains **4,646 clear pairs and 1,452 aggregate ties**, plus the separate 400-pair monitor. The old 4,846-clear export is preserved. Original 2,000-pair membership is unchanged. Public validation bytes are copied exactly from the old bundle; calibration/final-test files were not opened during this preparation.

| Mix | Previous monitor | New monitor | New clear training pool |
| --- | ---: | ---: | ---: |
| Pairs | 200 | 400 | 4,646 |
| Shorter winner | 48.0% | 41.25% | 41.28% |
| Strength 1 / 2 / 3 | 44.5 / 33.5 / 22.0% | 43.5 / 35.75 / 20.75% | 43.56 / 35.73 / 20.71% |
| Multi-turn | 32.5% | 28.75% | 28.89% |

This monitor represents the present clear training distribution, not an artificially balanced length distribution. If a later experiment adds ties, retain clear-label monitor loss for comparability and preregister a separate soft-target diagnostic rather than quietly redefining the selection metric.

## Audit of the proposed explanations

The three final learning-rate checkpoints average to **150/200 (75.0%)** on the old monitor, versus 145/200 (72.5%) for the selected single checkpoint. Ensemble log-loss is **0.523612**. This reuses selection data and is not a full-validation ensemble result; three-to-five-model inference also costs proportionally more.

The 2e-5 versus 1e-5 monitor log-loss difference is −0.000736, with a paired 95% bootstrap interval **[−0.0247, +0.0242]**. Against 5e-5 it is −0.005915 **[−0.0387, +0.0256]**. These readings do not clearly distinguish the rates; they do not prove exact equality either. Enlarging the monitor can improve precision but cannot guarantee reliable separation of very small differences.

On that monitor, selected shorter-winner accuracy is 55/96 (57.3%) and other-pair accuracy 90/104 (86.5%). Reweighting those two observed rates to validation's shorter-winner share gives **75.35%**, versus observed 72.5% and validation 81.0%. Length mix could explain part of the gap, not all of it; this is a descriptive calculation, not a causal finding.

## Fixed first data curve

Keep Qwen3-4B-Instruct-2507, the existing JSON prompt/tokenization, corrected FP32 A/B head, hard labels, unweighted loss, all 36 LoRA layers, rank 16/alpha 320/dropout .05, clipping 1.0, bias-corrected Adam and seed 20261007. Use fresh adapters for each size. Rate stays 2e-5; warmup stays 50 updates, then cosine decay over that candidate's planned one-pass updates. This measures the practical effect of more unique pairs and the corresponding extra updates, not a fixed-compute causal experiment.

| Clear pairs | Updates for one pass | 400-pair monitor checks | Estimated training + monitors |
| --- | ---: | ---: | ---: |
| 1,200 | 300 | 3 | 59 minutes |
| 2,400 | 600 | 6 | 117 minutes |
| 4,646 | 1,162 | 12 | 229 minutes |

The last update keeps the final two pairs; none is dropped. Every monitor reads both answer orders. Save the best checkpoint by monitor log-loss, record both its actual exposure and the final trained exposure, and report the final checkpoint as a secondary diagnostic if selection picks an early version. A best checkpoint selected before all pairs were seen must not be described as having learned from the full pool.

After all candidates complete, choose the lowest monitor log-loss; exact ties prefer the smaller pool, then earlier update. Full validation follows that selection. Its one planned primary contrast is against the frozen corrected 600-pair judge (81.0%, log-loss 0.450), using the unchanged +3-point accuracy rule or .02 log-loss rule, paired intervals and McNemar. Keep unchanged Qwen (76.5%) as a secondary reference. The two adjacent-size changes are exploratory data-curve diagnostics, with their paired intervals reported. Do not select the winner from validation accuracy. This prevents silently treating three separately tested validation winners as one prespecified comparison.

For each full evaluation, retain all 396 pairs/two orders, 311 clear labels as the main grade and 85 ties diagnostic. Reuse the 10,000 question-group bootstrap convention. Report accuracy, log-loss, AUC, shorter-winner and strength slices, order agreement and coverage. The historical reference used a different monitor; only the three new candidates share the enlarged selection procedure. Neither that reference contrast nor reused validation certifies a final score.

Do the one-pass curve first. A second-pass experiment changes training duration and must retain the same monitor and use seeded reshuffling plus patience. Do not mix one pass at one size and two at another and call the result a pure data-size comparison.

## Budget and execution gate

Historical L4 updates averaged **8.98 seconds per update**. Conservatively attributing all remaining sweep time to monitoring and doubling its size gives about **276 seconds per 400-pair monitor**. This estimates **6.75 GPU-hours** for the three one-pass training jobs, before setup/probe/full-validation work. Different sequence lengths and checkpoint writes can change timing. Two complete passes would take roughly twice the work; they are not included in a 5–6-hour estimate.

**Proposed all-three ceiling: eight GPU-hours, pending a budget choice.** Bound each job to 75/145/275 minutes and also the remaining overall budget; the per-job ceilings do not add extra time to that overall cap. Allow at most 20 minutes for a fresh probe and at most 20 minutes per full validation, still within the same overall ceiling. Stop on a failure or exhausted budget, preserve partial evidence and do not retry automatically. A six-hour budget should be staged rather than silently exceeded. No GPU job, runtime reconnection, automation or purchase was started for this preparation.

The changed trainer and new bundle require a new source/data fingerprint and a fresh CUDA probe. Old checkpoints and the reusable historical upload ZIP must keep their original code/data; do not attempt to resume them with these changed sources. The current deliverable is source, a versioned local bundle, passing CPU checks and the [machine-readable plan](results/data_curve_preparation_20261007/experiment_plan.json). A [portable package](results/data_curve_preparation_20261007/Judge_Data_Curve_Package.zip) has also been built and fingerprint-verified; it has not been uploaded. Fresh notebook setup and live launch follow the budget choice and must retain these settings.

## Steps after the data curve

1. **Length balancing:** rerun the chosen data setting with only `--balance-length` changed. Keep 80% shorter-winner accuracy as an aspiration, not a promised outcome; inspect its 119-pair uncertainty and overall log-loss/accuracy regressions.
2. **Soft targets:** compare hard labels with `--labels votes` on identical clear pairs first. Add all 1,452 ties with `--pool all_votes` only as a separate data comparison. A smoothing control uses existing 0.95/0.05 targets. Keep strength weighting a separate run; its new flag is not enabled automatically.
3. **Bigger models:** a flat curve can also reflect labels, optimization, input format or mismatch between data and evaluation; it does not prove a size bottleneck. Qwen3-8B/14B support thinking and non-thinking modes. A decision-token judge needs an explicit non-thinking template, fresh token IDs and a checked decision boundary. Current loader/adapter-layer validation remains restricted to the tested 4B architecture. Generalizing it, exporting with the new tokenizer and probing memory are a separate milestone; quantization is another measured change. [8B card](https://huggingface.co/Qwen/Qwen3-8B), [14B card](https://huggingface.co/Qwen/Qwen3-14B).
4. **More data:** HelpSteer3-Preference has 40,476 samples across its provided train/validation splits, not 40,000 new training pairs automatically available. Audit overlap against all project holdouts, including the permanent monitor, before importing. An automated audit may compare reserved split signatures without revealing their examples or scoring models on them. Record the retained count and licenses. A mixed-data run followed by a HelpSteer2-only pass is a later controlled experiment. [NVIDIA dataset card](https://huggingface.co/datasets/nvidia/HelpSteer3/blob/main/README.md?code=true).
5. **Ensembles:** test fixed seeds/data orders and equal probability averaging after individual runs work. Select ensemble membership on the monitor, then compare at both equal model count and equal inference budget. Do not optimize combinations repeatedly on validation.
6. **Lock, calibrate, test once:** freeze model, prompt, averaging and abstention policy before calibration fitting. Then evaluate the reserved 769-row test once. Previously recorded incidental calibration-text exposure stays in the audit; no 90% performance claim is supported by the reused development validation.

The order remains: **reliable measurement → data curve → controlled bias/label changes → model/data scaling if supported → final calibration/test**. These changes train the judge; the product and optional foundation-model learning project remain separate.
