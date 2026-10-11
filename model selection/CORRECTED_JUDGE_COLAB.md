# Corrected judge training: prepare Colab, then measure

**New preparation:** the completed run below remains historical. Current editable trainer sources now support the [data-curve plan](DATA_CURVE_PLAN.md); the original upload ZIP and frozen Drive source snapshot remain unchanged. Do not mix the new bundle or source with an old checkpoint resume.

Updated October 7, 2026. **The corrected Colab milestone is complete:** all tests, the real GPU probe, three-rate sweep, six training-side monitors, checkpoint selection and both full validations passed. Selected `2e-5`, update 150: **81.0% clear accuracy versus 76.5%**, log-loss **0.450 versus 1.169**. The predefined probability-quality rule supports a longer 2,000-pair experiment; it has not started. [Full measured report](CORRECTED_JUDGE_COLAB_RESULTS.md).

**Planned scope completed:** the run continued automatically through the longer-training decision. Saved reports, source/data verification and the local audit passed. Do not rerun setup or launch cells. Checkpoints remain on Drive. The completed Mac sweep remains historical evidence.

## Instructions for a fresh experiment, step by step

1. Open [Judge_Colab.ipynb](colab/Judge_Colab.ipynb) in Google Colab using its notebook upload option.
2. Select a GPU runtime. The code requires CUDA with BF16 support and checks it; L4/A100 are examples, not promised availability.
3. Run the upload cell and choose [Judge_Colab_Package.zip](colab/Judge_Colab_Package.zip). It contains public tokenized data and code, without model weights or credentials.
4. Run the setup cell: it removes unused torchvision/torchaudio/torchao add-ons and installs the pinned text dependencies. Choose **Runtime → Restart session**, then continue at the tests/Drive cell. That cell restores the paths. Repeating upload now verifies and reuses an existing intact package.
5. Mount Google Drive in the notebook so checkpoints survive a runtime disconnect. The default output folder is `MyDrive/judge_corrected_20261007`; use a new name for a new experiment.
6. Set `RUN_PROBE = True` and run that cell. Its 20-minute limit includes downloading Qwen and checking two updates, a long input, memory, and exact restart. Read the resulting report/log before continuing.
7. If the probe passes, inspect its timings and set `RUN_SWEEP = True`. The default ceiling is two hours per rate, six hours total; this is a limit, not an ETA. The sweep stops on failure without retrying.
8. After selection finishes, set `RUN_FULL_VALIDATION = True`. This measures unchanged Qwen and the selected adapter on all 396 pairs, with the same corrected head. Each evaluation has a one-hour limit. Open `selected_fp32/full_report.json` in Drive.
9. Review that result before starting longer training. Disconnect the GPU runtime when finished.

All three execution flags start as `False`, so “Run all” does not launch a sweep. The controlled notebook session completed the probe, sweep and both full validations. The directions above are for a fresh run; do not repeat them in the active session.

## Repair an existing Colab session

The reported `torchvision::nms` error occurs before either failing test loads Qwen. The Qwen module error is a wrapper around the image-package failure. The setup pinned PyTorch but did not account for Colab's preinstalled image/audio extensions; exact remote package versions were not included in the traceback. PyTorch documents that the compiled Torch/torchvision packages must be compatible. [PyTorch explanation](https://discuss.pytorch.org/t/runtimeerror-operator-torchvision-nms-does-not-exist/192829/6).

This unquantized text judge does not use torchvision, torchaudio or torchao. Run the following **once in a new Colab cell**, keeping the installed core packages:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", "torchao"], check=True, timeout=120)
```

Next choose **Runtime → Restart session**, then rerun the existing tests/Drive cell. Do not use “Disconnect and delete runtime,” repeat the upload cell, or skip the tests. A session restart clears cached imports while retaining the extracted files. If the tests still fail, their full output remains visible and GPU work stays blocked. All nine tests now pass in the actual Colab installation (8.073 seconds). This includes tiny-model restart tests; the full-model GPU probe is a separate gate.

The refreshed notebook includes this cleanup in setup and prints both test output streams before checking the exit code. The training scripts, package pins, model recipe, input data and source fingerprints used by workers are unchanged. An existing extracted package can therefore use the repair cell without another upload. [Saved failure evidence](results/colab_setup_fix_20261007T205148504717Z/failure_evidence.json).

## What was fixed first

A **gradient** tells training which direction to change a learned number. Gradient clipping limits the total size of one update's accumulated signal. The trainer averages both answer orders of four pairs, measures the global norm, clips it to **1.0**, then calls Adam. `raw_gradient_norm` records its size before clipping. The final partial batch of a full-data pass retains both orders of its remaining pairs and divides by its actual presentation count; no pair is silently dropped.

**Adam** is the optimizer that turns gradients into updates. Local MLX now enables `bias_correction=True`; PyTorch Adam applies its bias correction. This changes the recipe, so the previous winning rate `2e-5` must be checked again. Old optimizer states are not silently reused in the corrected MLX recipe. [PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html) and [global gradient clipping](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html).

A **logit** is a raw score before conversion into a probability. Training and evaluation now cast the final hidden vector and the A/B output rows to **float32 before multiplying**. Casting a rounded BF16 result afterwards cannot recover lost differences. New binary inference computes just rows 32 and 33. The local historical diagnostic helper can still request the complete float32 vocabulary projection; new Colab scoring always requests the two rows.

The **selection monitor** chooses a checkpoint during training. It now contains 200 clear pairs from 200 training groups outside the unchanged 2,000-pair pool. Those groups are excluded from every exported training mode, including later tie training. Validation remains 396 pairs: 311 clear labels and 85 ties. Moving the monitor protects future checkpoint selection; it cannot make the already-inspected validation set untouched again.

## What is in the package

| Part | Purpose |
| --- | --- |
| `bundle/training.jsonl` | 6,298 eligible pairs outside the monitor: 4,846 clear and 1,452 aggregate ties |
| `bundle/monitor.jsonl` | 200 clear training-side pairs, excluded from updates |
| `bundle/validation.jsonl` | All 396 validation pairs; not opened by the training worker |
| `bundle/manifest.json` | File fingerprints, original pool order, model revision, group identities and data counts |
| `code/probe.py`, `launch.py` | CUDA readiness check and an external wall-clock deadline |
| `code/train.py`, `sweep.py` | Paired training, explicit restart, and the corrected three-rate comparison |
| `code/evaluate.py`, `report.py`, `metrics.py` | Full evaluation and the established grouped uncertainty calculations |
| `code/protocol.py`, `test_colab.py` | Label handling, split checks and nine local CPU tests |

The local [exporter](colab/export_bundle.py) reads only training and validation. Both JSON and readable text are tokenized in advance; individual votes, labels and annotator justifications stay out of prompts. Training uses a common 2,048-token limit across formats, without truncation. Evaluation permits 4,096 tokens. The complete original 2,000-pair pool still fits. Calibration and final test are absent from the package.

The frozen Qwen model uses BF16 weights. LoRA—the small trainable additions—uses rank 16, dropout 0.05, and seven projections in each of 36 layers. The old MLX scale was 20; PEFT uses `alpha/r`, so **alpha is 320**, not 20. Adapter weights and Adam state use float32. Initialization and numerical kernels differ across MLX and PyTorch, so this is not a claim of bit-for-bit backend equivalence. [PEFT LoRA configuration](https://huggingface.co/docs/peft/main/en/package_reference/lora).

The local PyTorch environment is separate from the protected MLX environment. Tested imports: Torch 2.14.1, Transformers 5.19.0, PEFT 0.21.2. The notebook pins these and supporting packages. The actual L4 probe and completed jobs verified installation, compatibility, memory and runtime for this session; a fresh session must pass its own probe. The code stops rather than silently changing to quantization or FP16.

## Preregistered plan used for this completed experiment

**Hypothesis:** the corrected training recipe can improve probability quality against unchanged Qwen using the same CUDA/float32 scorer. The first sweep changes only peak learning rate: `1e-5`, `2e-5`, `5e-5`. All rates begin from the same seeded initialization, see the same first 600 pairs of the 2,000-pair pool, train 150 updates, warm up for 50 updates, then decay with a cosine schedule. Hard labels, JSON and no length weighting stay fixed.

Measure the 200-pair monitor at updates 100 and 150. Require every rate to complete. Select the lowest monitor log-loss; exact ties prefer the lower rate, then earlier update. Full validation follows selection, and remeasures the unchanged base with the same FP32 head. Baseline v2's 239/311 result remains locked history; a head/backend change must be named in every new table row.

Keep the earlier development screen: either accuracy improves at least 3 percentage points, its paired interval excludes zero, McNemar p < .05, and log-loss does not increase; or log-loss decreases at least .02, its paired interval is below zero, and observed accuracy does not fall. The report uses 10,000 question-group bootstrap samples. This is one planned primary comparison; later ablations need their own recorded comparison family. All repeated validation development remains exploratory.

## What is supported but not run

1. **More clear data:** `--pool pool2000 --updates 500 --epochs 1`, or 1,000 updates/two passes, after choosing a corrected rate and measured budget. Use `--patience 2 --minimum-improvement .002` to stop if two monitor readings fail to improve by .002. Select the best monitor checkpoint, not necessarily the last. Full-data modes keep the same held-out groups permanently excluded.
2. **Targets:** `--labels hard`, `smooth` (0.95/0.05), or `votes` (fraction of negative signed votes = probability A is preferred). First compare all three on the same clear pool. `--pool all_votes --labels votes` adds ties and is a separate data experiment. `all_clear` excludes them. No variant is declared better yet.
3. **Length weighting:** `--balance-length` gives equal total training weight to shorter-winner and longer-winner clear pairs. Equal-length pairs and aggregate ties retain weight 1. The shorter-winner slice remains visible in every full report.
4. **Format and layer experiments:** `--style plain` and `--layers 16` are separate options. They require their own recorded plan/probe; the provided sweep deliberately uses JSON/all 36 layers. Readable formatting after training and top-16 performance are unmeasured.
5. **Confidence:** reports include accuracy versus retained coverage with equal-confidence boundary pairs kept together. These are uncalibrated diagnostics. Temperature scaling and the abstention threshold wait for a locked model and calibration data. The final test remains reserved.

Checkpoints store adapter values, Adam state, random state and file fingerprints. Explicit `--resume` requires the latest complete checkpoint, original arguments, matching source/data/environment, and an unfinished run. Interrupted logs and partial checkpoint/monitor files are moved into an `interrupted_…` folder, preserving evidence; the saved state determines where training restarts. `launch.py` preserves cumulative elapsed time and enforces the external deadline. Resume does not extend a budget. Retaining every short-sweep checkpoint requires roughly 10 GB of Drive space; original model downloads use separate runtime storage. No worker retries automatically. A whole failed sweep needs a reviewed recovery plan; the sweep launcher does not automatically resume it.

## What the checks actually established

- The existing 16 offline MLX/metrics/sweep tests passed before the real probe.
- Corrected real-Qwen MLX: two accumulated updates and both 2,045-token stress gradients passed at **10.216 GiB** peak. Fresh-process restart matched adapter, optimizer and random-state fingerprints exactly. This checks engineering behavior, not judge quality.
- Nine new CPU checks use a tiny actual Qwen architecture and PEFT. They cover float32 projection before rounding, clipping/Adam, target swaps including six votes, group isolation, balanced weights, known metrics, exact restart with dropout/evaluation, and an interrupted worker's explicit resume.
- All **314 sealed sweep artifacts** still match their saved fingerprints. [Audit](results/colab_preparation_20261007/correction_audit.json).
- The full-model CUDA probe passed (86.3 seconds including startup/download; 8.993 GiB peak allocated memory; exact restart). The corrected sweep is complete and selected `2e-5` at update 150. Full FP32 validation is **complete** and passes the probability-quality screen; longer training, calibration fitting, final-test evaluation and integration remain **not run**.

The audit confirms the large gradient spikes (maximum 3,287, median 58.70; 17 of 19 above 500 occurred in the first ten updates). It finds **52**, not 31, distinct A-minus-B differences and 24 exact equal-logit reads. Four clear pairs averaged exactly .5 and were counted as misses, but their corrected winners are not known until re-scoring. The claim that training mainly learned confidence or conversation formatting is a hypothesis, not an isolated causal finding.

Of 6,623 training pairs, 1,285/5,147 clear pairs have dissent (25.0%), 2,237/5,147 have strength 1 (43.5%), and 3,024/5,147 have a longer winner (58.8%). There are **17 six-vote pairs**. Only 134 of 1,476 aggregate ties have exactly half of votes for A; most have one-third or two-thirds. Soft vote targets preserve these differences instead of assuming every aggregate tie means 50/50. [Data audit](results/colab_preparation_20261007/bundle/data_audit.json).

## Live setup evidence

The second upload was renamed to `Judge_Colab_Package (1).zip`; the old exact-name assertion rejected it. The existing extracted package passed all 19 file fingerprints, so it was safely reused. Setup now handles the returned filename and verifies an existing folder instead of replacing it. [Observed failures and checks](results/colab_live_setup_20261007T210904Z/observations.json). Probe files were copied from `/content/judge_runs/probe_20261007T210824Z` into `MyDrive/judge_corrected_20261007/probe` before training began. Exact small evidence is also saved locally and verified. The first Drive mount returned `credential propagation was unsuccessful`; browser authorization is separate from package setup.

Earlier live observations captured the first rate complete and the second at update 62. Those saved progress snapshots remain history; all three rates and both full evaluations are now complete.

## Full validation completed — October 7 at 22:54 UTC

The sequential driver (PID 34451) ran the existing 60-minute launcher for unchanged Qwen, then the selected checkpoint, and exited successfully without retry. Every model scored 396 pairs in both orders. Local reconstruction verified all metrics, intervals and the unchanged screen. See [the full measured report](CORRECTED_JUDGE_COLAB_RESULTS.md) and [exact evidence/helper guides](results/colab_validation_20261007T224541Z/README.md). The notebook status and result cells show completion; do not rerun launch cells.
