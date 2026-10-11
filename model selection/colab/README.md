# Current notebook: one full-data A100 run

Open **Judge_A100_FullData.ipynb** and upload **Judge_A100_FullData_Package.zip** when its first cell asks. Choose A100; run cells in order. The notebook creates an isolated Python environment, passes tests and a GPU probe, then automatically runs one 4,646-pair training pass and both full validation evaluations. No manual training chunks or learning-rate sweep. [Exact recipe, limits and instructions](../A100_FULL_DATA.md).

Seventeen local and Colab tests passed. The A100 full-model probe passed and the single full-data training job is running; no new accuracy is available yet. The six-hour compute ceiling excludes the separately bounded 20-minute setup. Drive saves every 100 updates and at the end; existing outputs are never overwritten. The `--checkpoint-every` default remains 25 for other uses. Current sources and this new ZIP include the full-data supervisor. Historical notebooks/ZIPs and completed GPU records below remain unchanged.

## Historical corrected-sweep notebook

Open `Judge_Colab.ipynb` in Google Colab, select a BF16-capable CUDA GPU, and upload `Judge_Colab_Package.zip`. Follow the notebook in order: dependencies → tests → Drive → probe → sweep → full validation. All GPU-work flags begin as False. The notebook contains the limits and commands.

The original Colab experiment completed all three rates and full validation: selected clear accuracy 81.0%, log-loss 0.450. Its frozen upload ZIP and Drive source snapshot remain preserved. **Current editable sources have additional preparation changes** for [the data curve](../DATA_CURVE_PLAN.md): seeded later-epoch shuffling, `--max-pairs`, `--seed`, optional `--weight-strength`, `--minimum-epochs` and `prepare_curve.py`. Their twelve CPU tests passed; their own fresh CUDA probe has not run. The old upload ZIP does not include these new features. Do not resume historical jobs with the changed files.

The package contains public NVIDIA HelpSteer2 preference data under CC BY 4.0, revision `990b2711a36180dd19d9c94b8627844866f8982a`; source: https://huggingface.co/datasets/nvidia/HelpSteer2. Rows are transformed into token IDs, vote proportions and diagnostic metadata. It excludes calibration, final test, model weights, credentials and annotator prose.

`export_bundle.py` is a local preparation helper requiring the parent project and its prepared dataset; do not run it in Colab. The notebook uses its completed export.

The first cloud run must pass the bounded probe. Then compare the three corrected Adam rates using a training-side selection monitor. Re-evaluate unchanged Qwen and the selected adapter with the same FP32 head. New backend scores must not silently replace the locked baseline-v2 scores.

Longer training, soft labels, length balancing, top-16 layers, formatting after training, temperature scaling and final-test use remain later experiments. Only manual execution uses the Colab runtime; no automatic retry or hidden schedule is included. Checkpoints go to the chosen Drive folder. Disconnect the GPU when finished.

## Colab setup repair: torchvision::nms

If the tests fail while importing Qwen with `RuntimeError: operator torchvision::nms does not exist`, run this once in a new Colab cell:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", "torchao"], check=True, timeout=120)
```

These are optional image/audio/quantization add-ons, unused by this unquantized text judge. Colab also contained torchao 0.10.0, which caused two tests to fail under the pinned PEFT installation; removing it allowed all nine tests to pass. Then choose **Runtime → Restart session**, and rerun the tests/Drive cell. Do not delete the runtime or re-upload the package; `/content/judge_colab` should still be present. Do not skip failed tests or enable the sweep before the GPU probe passes. The refreshed notebook performs this cleanup during setup and displays both test output streams. The training scripts, data and pinned core versions are unchanged. Both observed import failures are recorded. All nine tests passed in the actual Colab runtime in 8.073 seconds. That verifies the test gate, not full-model GPU readiness or judge accuracy.

## Uploading again

The upload cell now accepts the filename returned by Colab, including a suffix such as `(1)`. If `/content/judge_colab` already exists, it verifies every manifest fingerprint and reuses that package without overwriting it. A mismatch stops setup for inspection. The package contains no credentials; Google Drive authorization remains a separate browser step.
