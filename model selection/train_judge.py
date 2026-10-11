"""Binary LoRA primitives and a bounded readiness runner; no sweep starts here."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from baseline import read_rows, save_json
from baseline_guard import sha256
from judge_v2 import HERE, first_logits, probabilities
from judge_head import project_hidden
from train_trial import adapter_digest


def schedule_pairs(pairs):
    if not pairs or len(pairs) % 4 or len({r["id"] for r in pairs}) != len(pairs):
        raise ValueError("Use distinct pairs in groups of four per update.")
    return [[{"id": row["id"], "group_id": row["group_id"], **row["orders"][order]}
             for i, row in enumerate(pairs[start:start + 4]) for order in ((0, 1) if i % 2 == 0 else (1, 0))]
            for start in range(0, len(pairs), 4)]


def decision_scores(model, tokens):
    import mlx.core as mx
    if getattr(model, "model_type", None) == "qwen3":
        return project_hidden(model, model.model(tokens)[:, -1, :])
    else:
        logits = model(tokens)[:, -1:, :]
        if logits.dtype != mx.float32:
            raise ValueError("Non-Qwen test models must supply float32 logits.")
    return logits[:, 0, :][:, mx.array([32, 33])].astype(mx.float32)


def binary_loss(model, tokens, target):
    import mlx.core as mx
    import mlx.nn as nn
    if target not in (0, 1) or tokens.shape[0] != 1 or tokens.shape[1] == 0:
        raise ValueError("One nonempty prompt and an A/B target are required.")
    scores = decision_scores(model, tokens)
    return nn.losses.cross_entropy(scores, mx.array([target]), reduction="mean")


def make_optimizer(config):
    import mlx.optimizers as optim
    peak, warm, total = (config[k] for k in ("learning_rate", "warmup_updates", "schedule_updates"))
    if not 0 < peak <= 5e-5 or not 1 < warm < total - 1:
        raise ValueError("Learning rate or warm-up schedule is outside the declared bounds.")
    ramp = optim.linear_schedule(peak / warm, peak, warm - 1)
    decay = optim.cosine_decay(peak, total - warm - 1, end=peak * config["final_lr_fraction"])
    if config.get("adam_bias_correction") is not True or config.get("gradient_clip_norm") != 1.0:
        raise ValueError("Use a new plan declaring bias-corrected Adam and norm-1 clipping.")
    return optim.Adam(learning_rate=optim.join_schedules([ramp, decay], [warm]), bias_correction=True)


def install_dropout(model, seed):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    class KeyedDropout(nn.Module):
        def __init__(self, keep):
            super().__init__()
            self._keep, self._key = keep, None
        def __call__(self, x):
            if not self.training or self._keep == 1:
                return x
            if self._key is None:
                raise ValueError("Assign this presentation's dropout keys before computing gradients.")
            mask = mx.random.bernoulli(self._keep, x.shape, key=self._key)
            return mask * x * (1 / self._keep)
    replacements = [(name, KeyedDropout(module._p_1)) for name, module in model.named_modules() if isinstance(module, nn.Dropout)]
    model.update_modules(tree_unflatten(replacements))
    model._judge_dropouts = [module for name, module in replacements]
    model._judge_random = [mx.random.key(seed)]


def refresh_dropout(model):
    import mlx.core as mx
    keys = mx.random.split(model._judge_random[0], len(model._judge_dropouts) + 1)
    model._judge_random[0] = keys[0]
    for index, module in enumerate(model._judge_dropouts, 1):
        module._key = keys[index]


def validate_inputs(output):
    plan = json.loads((output / "plan.json").read_text())
    for key in ("source_sha256", "reused_source_sha256"):
        for name, expected in plan[key].items():
            if sha256(HERE / name) != expected:
                raise ValueError("Code changed after planning: " + name)
    for name, expected in plan["files_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError("Prepared input changed: " + name)
    return plan


def load_training(config):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    mx.set_memory_limit(config["memory_limit_gib"] * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(config["seed"])
    model, _ = load(str(HERE / config["model"]), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    install_dropout(model, config["seed"])
    arrays = tree_flatten(model.trainable_parameters())
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    if config["num_layers"] not in (16, 36):
        raise ValueError("Only the all-36 control and top-16 experiment are supported.")
    expected = {f"model.layers.{i}.{p}.lora_{suffix}" for i in range(36 - config["num_layers"], 36) for p in projections for suffix in ("a", "b")}
    if len(model.layers) != 36 or {name for name, value in arrays} != expected:
        raise ValueError("The adapted block/projection names differ from the declared recipe.")
    grad_checkpoint(model.layers[0])
    model.train()
    optimizer = make_optimizer(config)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.trainable_parameters(), optimizer.state, model._judge_random)
    return model, optimizer, {"trainable_parameters": sum(value.size for name, value in arrays),
                              "trainable_arrays": sorted(expected), "projection_modules": len(expected) // 2}


def state_digests(model, optimizer):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.eval(model.trainable_parameters(), optimizer.state, model._judge_random)
    return {"adapter": adapter_digest(tree_flatten(model.trainable_parameters())),
            "optimizer": adapter_digest(tree_flatten(optimizer.state)),
            "random": adapter_digest([(str(i), value) for i, value in enumerate(model._judge_random)])}


def save_checkpoint(path, model, optimizer, config, output):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    if path.exists():
        raise ValueError("Never overwrite a saved checkpoint.")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.mkdir(exist_ok=False)
    digests = state_digests(model, optimizer)
    mx.save_safetensors(str(temporary / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(temporary / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    mx.save_safetensors(str(temporary / "random.safetensors"), {str(i): value for i, value in enumerate(model._judge_random)})
    save_json(temporary / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": config["num_layers"], "lora_parameters": config["lora_parameters"]})
    step = int(optimizer.step.item())
    save_json(temporary / "resume.json", {"completed_updates": step, "next_presentation": step * 8,
              "plan_sha256": sha256(output / "plan.json"), "config": config, "state_digests": digests,
              "files_sha256": {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}})
    temporary.rename(path)
    return digests


def restore_checkpoint(path, model, optimizer, config, output):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    saved = json.loads((path / "resume.json").read_text())
    if saved["config"] != config or saved["plan_sha256"] != sha256(output / "plan.json"):
        raise ValueError("Resume settings or the saved data-order plan differ.")
    for name, expected in saved["files_sha256"].items():
        if sha256(path / name) != expected:
            raise ValueError("Checkpoint is incomplete or changed: " + name)
    model.load_weights(str(path / "adapters.safetensors"), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    optimizer.init(model.trainable_parameters())
    random = mx.load(str(path / "random.safetensors"))
    model._judge_random[:] = [random[str(i)] for i in range(len(random))]
    if saved["completed_updates"] != int(optimizer.step.item()) or saved["next_presentation"] != int(optimizer.step.item()) * 8:
        raise ValueError("Optimizer step and data cursor disagree.")
    if state_digests(model, optimizer) != saved["state_digests"]:
        raise ValueError("Restored numerical state differs from the checkpoint.")
    return saved["completed_updates"]


def checked_gradient(model, calculate, item, config):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    if len(item["tokens"]) > config["training_token_cap"]:
        raise ValueError("Input exceeds the training cap; truncation is forbidden.")
    refresh_dropout(model)
    loss, gradients = calculate(model, mx.array([item["tokens"]]), item["target"])
    flat = tree_flatten(gradients)
    if {name for name, value in flat} != {name for name, value in tree_flatten(model.trainable_parameters())}:
        raise ValueError("A gradient is missing or reaches a frozen base parameter.")
    mx.eval(loss, gradients, model._judge_random)
    norm = mx.sqrt(sum(mx.sum(value.astype(mx.float32) ** 2) for name, value in flat))
    mx.eval(loss, gradients, norm, model._judge_random)
    value, magnitude = float(loss.item()), float(norm.item())
    if not math.isfinite(value) or not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Require finite loss and finite, nonzero gradients.")
    if mx.get_peak_memory() / 1024**3 > config["memory_limit_gib"]:
        raise ValueError(f"MLX peak {mx.get_peak_memory() / 1024**3:.3f} GiB exceeded the guideline at {len(item['tokens'])} tokens.")
    return value, gradients, magnitude


def update(model, optimizer, calculate, batch, config):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    if len(batch) != config["examples_per_update"] or len(batch) != 8:
        raise ValueError("An update must contain exactly eight presentations.")
    started, accumulated, readings = time.perf_counter(), None, []
    for item in batch:
        tick = time.perf_counter()
        loss, gradients, norm = checked_gradient(model, calculate, item, config)
        accumulated = tree_map(lambda g: g / 8, gradients) if accumulated is None else tree_map(lambda a, g: a + g / 8, accumulated, gradients)
        mx.eval(accumulated)
        readings.append({"id": item["id"], "swapped": item["swapped"], "target": item["target"],
                         "tokens": len(item["tokens"]), "loss": loss, "gradient_norm": norm, "seconds": time.perf_counter() - tick})
        del gradients
        mx.clear_cache()
    accumulated = tree_map(lambda g: g.astype(mx.float32), accumulated)
    accumulated, raw_norm = optim.clip_grad_norm(accumulated, max_norm=1.0)
    mx.eval(accumulated, raw_norm)
    raw_norm = float(raw_norm.item())
    if not math.isfinite(raw_norm):
        raise ValueError("Accumulated gradient norm must be finite before the update.")
    optimizer.update(model, accumulated)
    mx.eval(model.trainable_parameters(), optimizer.state)
    from mlx.utils import tree_flatten
    if not all(bool(mx.all(mx.isfinite(a)).item()) for name, a in tree_flatten(model.trainable_parameters())):
        raise ValueError("An update produced non-finite adapter values.")
    if mx.get_peak_memory() / 1024**3 > config["memory_limit_gib"]:
        raise ValueError("The memory guideline was exceeded after the update.")
    return {"step": int(optimizer.step.item()), "learning_rate": float(optimizer.learning_rate.item()),
            "raw_gradient_norm": raw_norm, "gradient_clip_norm": 1.0,
            "mean_loss": sum(r["loss"] for r in readings) / 8, "presentations": readings,
            "seconds": time.perf_counter() - started, "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3}


def worker(output, phase):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    plan = validate_inputs(output)
    if phase == "checks":
        from test_train_judge import gpu_checks
        save_json(output / "checks.json", gpu_checks(plan["config"]))
        return
    if json.loads((output / "checks.json").read_text())["status"] != "passed":
        raise ValueError("The small mathematical checks must pass first.")
    import mlx.core as mx
    import mlx.nn as nn
    model, optimizer, architecture = load_training(plan["config"])
    destination = output / phase
    destination.mkdir(exist_ok=False)
    save_json(destination / "architecture.json", architecture)
    before = state_digests(model, optimizer)
    batches = schedule_pairs(read_rows(output / "probe_pairs.jsonl"))
    if phase == "continuous":
        model.eval()
        tokens = mx.array([batches[0][0]["tokens"]])
        full = project_hidden(model, model.model(tokens)[0, -1], token_ids=None)
        final = decision_scores(model, tokens)[0]
        p_full = float(mx.softmax(full[mx.array([32, 33])])[0].item())
        p_final = float(mx.softmax(final)[0].item())
        if abs(p_full - p_final) > plan["acceptance"]["scoring_probability_tolerance"]:
            raise ValueError("Final-position head optimization changed A/B probability beyond tolerance.")
        save_json(destination / "head_check.json", {"p_full": p_full, "p_final_only": p_final, "absolute_difference": abs(p_full - p_final)})
        del tokens, full, final
        mx.clear_cache()
        model.train()
    start = restore_checkpoint(output / "continuous/checkpoint_0001", model, optimizer, plan["config"], output) if phase == "resumed" else 0
    calculate = nn.value_and_grad(model, binary_loss)
    with (destination / "updates.jsonl").open("x") as stream:
        for batch in batches[start:]:
            entry = update(model, optimizer, calculate, batch, plan["config"])
            save_checkpoint(destination / f"checkpoint_{entry['step']:04d}", model, optimizer, plan["config"], output)
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            print(json.dumps({k: v for k, v in entry.items() if k != "presentations"}), flush=True)
    after = state_digests(model, optimizer)
    if before["adapter"] == after["adapter"]:
        raise ValueError("No adapter update occurred.")
    stress = []
    if phase == "continuous" and plan["run_long_stress"]:
        for item in read_rows(output / "longest_pair.jsonl")[0]["orders"]:
            tick = time.perf_counter()
            loss, gradients, norm = checked_gradient(model, calculate, item, plan["config"])
            stress.append({"tokens": len(item["tokens"]), "swapped": item["swapped"], "loss": loss, "gradient_norm": norm,
                           "seconds": time.perf_counter() - tick, "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3})
            del gradients
            mx.clear_cache()
        if state_digests(model, optimizer)["adapter"] != after["adapter"] or int(optimizer.step.item()) != 2:
            raise ValueError("The dry stress probe changed weights or optimizer steps.")
        model.eval()
        ids = batches[0][0]["tokens"]
        direct = decision_scores(model, mx.array([ids]))[0]
        cached = first_logits(model, ids, 512, token_ids=(32, 33))
        p_direct = float(mx.softmax(direct)[0].item())
        p_cached = float(mx.softmax(cached)[0].item())
        if abs(p_direct - p_cached) > plan["acceptance"]["scoring_probability_tolerance"]:
            raise ValueError("Training-position and v2 inference probabilities disagree.")
        save_json(destination / "scoring_check.json", {"p_direct": p_direct, "p_v2_cached": p_cached, "absolute_difference": abs(p_direct - p_cached)})
    save_json(destination / "result.json", {"status": "passed", "before": before, "after_updates": after,
              "completed_updates": int(optimizer.step.item()), "stress_probes": stress,
              "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3, "base_weights_written": False})


def supervise(output, phase):
    plan = validate_inputs(output)
    if phase not in plan["phase_wall_seconds"] or (output / (phase + "_worker.json")).exists():
        raise ValueError("Use an unstarted phase; failed runs are preserved, not retried automatically.")
    limit = plan["phase_wall_seconds"][phase]
    result = {"status": "running", "phase": phase, "wall_limit_seconds": limit}
    save_json(output / (phase + "_worker.json"), result)
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(output), phase], timeout=limit, check=False)
        result["status"] = "completed" if process.returncode == 0 else "worker_failed"
        result["returncode"] = process.returncode
    except subprocess.TimeoutExpired:
        result["status"] = "deadline_exceeded"
    result["seconds"] = time.perf_counter() - started
    save_json(output / (phase + "_worker.json"), result)
    print(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        folder, phase = Path(sys.argv[2]).resolve(), sys.argv[3]
        try:
            worker(folder, phase)
        except Exception as error:
            save_json(folder / (phase + "_failure.json"), {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
            raise
    elif len(sys.argv) == 4 and sys.argv[3] == "--run":
        supervise(Path(sys.argv[1]).resolve(), sys.argv[2])
    else:
        raise SystemExit("Use train_judge.py RUN_FOLDER checks|continuous|resumed --run.")
