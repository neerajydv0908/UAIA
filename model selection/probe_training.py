"""Check LoRA gradients and memory without applying updates or saving a model."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline import save_json
from baseline_guard import ROOT, sha256, verify
from prepare_training import tokenize_record


def answer_only_loss(model, batch, lengths):
    import mlx.core as mx
    from mlx_lm.tuner.trainer import default_loss
    corrected = mx.stack([lengths[:, 0], lengths[:, 1] - 1], axis=1)
    return default_loss(model, batch, corrected)


def worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint, iterate_batches
    from mlx_lm.tuner.utils import linear_to_lora_layers
    guard = verify()
    toy = mx.array([[1, 2, 3, 4, 5, 0, 0]])
    toy_lengths = mx.array([[3, 5]])
    toy_model = lambda inputs: mx.zeros((*inputs.shape, 8))
    _, library_tokens = default_loss(toy_model, toy, toy_lengths)
    _, corrected_tokens = answer_only_loss(toy_model, toy, toy_lengths)
    mask_check = {"expected_target_tokens": 2, "library_default_tokens": int(library_tokens.item()), "corrected_tokens": int(corrected_tokens.item())}
    save_json(output / "padding_mask_check.json", mask_check)
    if mask_check["corrected_tokens"] != 2:
        raise ValueError("The corrected loss includes prompt or padding tokens.")
    profile = json.loads((output / "profile.json").read_text())
    for name, expected in profile["files_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError("Prepared training file changed: " + name)
    config = json.loads((output / "training_config.json").read_text())
    probes = json.loads((output / "probe_records.json").read_text())
    mx.set_memory_limit(config["mlx_memory_limit_gib"] * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(config["seed"])
    started = time.perf_counter()
    model, tokenizer = load(str(ROOT / "model selection" / config["model"]), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    load_seconds = time.perf_counter() - started
    model.freeze()
    linear_to_lora_layers(model, config["lora_layers"], config["lora_parameters"])
    trainable = tree_flatten(model.trainable_parameters())
    if len(trainable) != 16 or any(not name.endswith((".lora_a", ".lora_b")) for name, value in trainable):
        raise ValueError("Unexpected trainable parameters; base weights must remain frozen.")
    grad_checkpoint(model.layers[0])
    model.train()
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    optimizer.init(model.trainable_parameters())
    mx.eval(model.trainable_parameters(), optimizer.state)
    def adapter_hash():
        digest = hashlib.sha256()
        for name, value in tree_flatten(model.trainable_parameters()):
            digest.update(name.encode())
            digest.update(np.asarray(value).tobytes())
        return digest.hexdigest()
    before = adapter_hash()
    results = []
    calculate = nn.value_and_grad(model, answer_only_loss)
    for name, record in probes.items():
        tokens, offset = tokenize_record(record, tokenizer)
        if len(tokens) > config["max_seq_length"]:
            raise ValueError("Probe exceeds the cap; truncation is forbidden.")
        dataset = ChatDataset([record], tokenizer, mask_prompt=True)
        if dataset.process(record) != (tokens, offset):
            raise ValueError("MLX trainer tokenization or prompt mask differs from preparation.")
        batch, lengths = next(iterate_batches([(tokens, offset)], batch_size=1, max_seq_length=config["max_seq_length"], seed=config["seed"]))
        started = time.perf_counter()
        (loss, target_tokens), gradients = calculate(model, batch, lengths)
        mx.eval(loss, target_tokens, gradients)
        values = [value for key, value in tree_flatten(gradients)]
        finite = bool(mx.isfinite(loss).item()) and all(bool(mx.all(mx.isfinite(value)).item()) for value in values)
        norm_squared = sum(float(mx.sum(value.astype(mx.float32) ** 2).item()) for value in values)
        peak = mx.get_peak_memory() / 1024**3
        result = {"probe": name, "sequence_tokens": len(tokens), "padded_tokens": batch.shape[1], "prompt_tokens": offset,
                  "target_tokens": int(target_tokens.item()), "expected_target_tokens": len(tokens) - offset,
                  "loss": float(loss.item()), "finite": finite, "gradient_norm": norm_squared ** 0.5,
                  "seconds": time.perf_counter() - started, "mlx_peak_memory_gib": peak}
        results.append(result)
        save_json(output / "probe_progress.json", results)
        print(json.dumps(result), flush=True)
        if not finite or not 0 < norm_squared < float("inf") or result["target_tokens"] != result["expected_target_tokens"]:
            raise ValueError("The loss, gradients, or answer-only mask failed its check.")
        if peak > config["mlx_memory_limit_gib"]:
            raise ValueError("Observed memory exceeded the chosen guideline; do not proceed to training.")
        del gradients, values, loss, target_tokens, batch, lengths
        mx.clear_cache()
    after = adapter_hash()
    if before != after or optimizer.step.item() != 0:
        raise ValueError("The dry probe unexpectedly changed adapter weights or optimizer steps.")
    save_json(output / "probe_result.json", {"status": "passed", "baseline_guard": guard, "load_seconds": load_seconds,
              "padding_mask_check": mask_check, "probes": results, "trainable_parameters": sum(value.size for key, value in trainable),
              "trainable_arrays": [key for key, value in trainable], "adapter_sha256_before": before, "adapter_sha256_after": after,
              "optimizer_updates": 0, "optimizer_state_allocated": True, "checkpoint_saved": False,
              "base_weights_written": False, "memory_limit_kind": "MLX guideline, not a hard whole-process memory cap"})


def main(output):
    if not (output / "profile.json").is_file():
        raise ValueError("Choose a completed training-readiness folder.")
    if any((output / name).exists() for name in ("probe_run.json", "probe_progress.json", "probe_result.json")):
        raise ValueError("This folder already has a probe; prepare a new run instead of overwriting it.")
    config = json.loads((output / "training_config.json").read_text())
    run = {"status": "running", "code_sha256": sha256(Path(__file__)), "wall_limit_seconds": config["probe_wall_seconds"], "optimizer_updates_planned": 0}
    save_json(output / "probe_run.json", run)
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(output)], check=False, timeout=config["probe_wall_seconds"])
        run["status"] = "completed" if process.returncode == 0 else "worker_failed"
    except subprocess.TimeoutExpired:
        run["status"] = "deadline_exceeded"
    run["wall_seconds"] = time.perf_counter() - started
    save_json(output / "probe_run.json", run)
    print(json.dumps(run, indent=2))
    if run["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 2:
        main(Path(sys.argv[1]).resolve())
    else:
        raise SystemExit("Use probe_training.py PATH_TO_PREPARED_RUN.")
