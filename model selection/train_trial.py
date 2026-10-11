"""Run one bounded LoRA trial, reload it, and compare with protected baseline v1."""
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from baseline import classify_output, messages_for, read_rows, save_json, summarize
from baseline_guard import ROOT, SNAPSHOT, sha256, verify
from prepare_training import tokenize_record
from probe_training import answer_only_loss

HERE = Path(__file__).resolve().parent
READY = HERE / "results/training_readiness_20261006T101929653463Z"


def schedule(index, updates, seed):
    if not 0 < updates <= 20 or updates % 2 or updates > 2 * len(index):
        raise ValueError("Use an even number of updates between 2 and 20, with enough groups.")
    if len({row["group_id"] for row in index}) != len(index):
        raise ValueError("Training groups must be distinct.")
    groups = random.Random(seed).sample(range(len(index)), updates // 2)
    return [{"record_index": 2 * group + int(swapped), "id": index[group]["id"],
             "group_id": index[group]["group_id"], "swapped": swapped}
            for order, group in enumerate(groups)
            for swapped in ((False, True) if order % 2 == 0 else (True, False))]


def check_ready(folder):
    profile = json.loads((folder / "profile.json").read_text())
    for name, expected in profile["files_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError("Prepared file changed: " + name)
    probe = json.loads((folder / "probe_result.json").read_text())
    config = json.loads((folder / "training_config.json").read_text())
    if probe["status"] != "passed" or probe["optimizer_updates"] != 0:
        raise ValueError("Require the completed dry probe with zero updates.")
    expected = {"batch_size": 1, "max_seq_length": 2048, "lora_layers": 4, "mask_prompt": True,
                "grad_checkpoint": True, "optimizer": "Adam", "future_trial_iterations": 20}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("This first trial must use the measured readiness settings.")
    if not 0 < config["future_trial_wall_seconds"] <= 900 or not 0 < config["learning_rate"] <= 0.00001:
        raise ValueError("Trial time or learning rate is outside the allowed bound.")
    return config, probe


def adapter_digest(arrays):
    import numpy as np
    digest = hashlib.sha256()
    for name, value in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def start_worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    plan = json.loads((output / "plan.json").read_text())
    verify()
    for name, expected in plan["code_sha256"].items():
        if sha256(HERE / name) != expected:
            raise ValueError("Experiment code changed after the plan was saved: " + name)
    if sha256(READY / "profile.json") != plan["training_profile_sha256"]:
        raise ValueError("Prepared profile changed after planning.")
    config, probe = check_ready(READY)
    if config != plan["config"]:
        raise ValueError("Prepared settings changed after planning.")
    return plan, config, probe


def train_worker(output):
    plan, config, probe = start_worker(output)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches
    from mlx_lm.tuner.utils import linear_to_lora_layers
    mx.set_memory_limit(config["mlx_memory_limit_gib"] * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(config["seed"])
    model, tokenizer = load(str(HERE / config["model"]), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    model.freeze()
    linear_to_lora_layers(model, config["lora_layers"], config["lora_parameters"])
    arrays = tree_flatten(model.trainable_parameters())
    if set(name for name, value in arrays) != set(probe["trainable_arrays"]):
        raise ValueError("Only the previously checked LoRA arrays may train.")
    before = adapter_digest(arrays)
    trainable_count = sum(value.size for name, value in arrays)
    del arrays
    records = read_rows(READY / "trial_train.jsonl")
    encoded = [tokenize_record(record, tokenizer) for record in records]
    if any(len(tokens) > config["max_seq_length"] for tokens, offset in encoded):
        raise ValueError("Training input exceeds the cap; truncation is forbidden.")
    grad_checkpoint(model.layers[0])
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    optimizer.init(model.trainable_parameters())
    model.train()
    calculate = nn.value_and_grad(model, answer_only_loss)
    adapter_config = {"fine_tune_type": "lora", "num_layers": config["lora_layers"], "lora_parameters": config["lora_parameters"]}
    with (output / "training_log.jsonl").open("x") as stream:
        for step, item in enumerate(plan["schedule"], 1):
            started = time.perf_counter()
            tokens, offset = encoded[item["record_index"]]
            batch, lengths = next(iterate_batches([(tokens, offset)], batch_size=1, max_seq_length=config["max_seq_length"], seed=config["seed"]))
            (loss, target_count), gradients = calculate(model, batch, lengths)
            mx.eval(loss, target_count, gradients)
            values = [value for name, value in tree_flatten(gradients)]
            finite = bool(mx.isfinite(loss).item()) and all(bool(mx.all(mx.isfinite(value)).item()) for value in values)
            if not finite or int(target_count.item()) != len(tokens) - offset:
                raise ValueError("Non-finite gradients/loss or incorrect target mask; update refused.")
            if mx.get_peak_memory() / 1024**3 > config["mlx_memory_limit_gib"]:
                raise ValueError("Memory guideline exceeded before update.")
            optimizer.update(model, gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            arrays = tree_flatten(model.trainable_parameters())
            if not all(bool(mx.all(mx.isfinite(value)).item()) for name, value in arrays):
                raise ValueError("An update produced non-finite adapter values.")
            if int(optimizer.step.item()) != step:
                raise ValueError("Optimizer step count disagrees with the planned update count.")
            entry = {**item, "step": step, "target": records[item["record_index"]]["messages"][-1]["content"],
                     "sequence_tokens": len(tokens), "target_tokens": int(target_count.item()), "loss_before_update": float(loss.item()),
                     "seconds": time.perf_counter() - started, "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3}
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            print(f"Update {step}/{len(plan['schedule'])}: loss={entry['loss_before_update']:.4f}, peak={entry['mlx_peak_memory_gib']:.2f} GiB", flush=True)
            if entry["mlx_peak_memory_gib"] > config["mlx_memory_limit_gib"]:
                raise ValueError("Memory guideline exceeded after update; trial stopped.")
            if step % 10 == 0:
                checkpoint = output / ("adapter" if step == len(plan["schedule"]) else f"checkpoint_{step:04d}")
                checkpoint.mkdir(exist_ok=False)
                save_json(checkpoint / "adapter_config.json", adapter_config)
                mx.save_safetensors(str(checkpoint / "adapters.safetensors"), dict(arrays))
            del gradients, values, arrays, loss, target_count, batch, lengths
            mx.clear_cache()
    after = adapter_digest(tree_flatten(model.trainable_parameters()))
    if before == after:
        raise ValueError("No adapter changes were detected after training.")
    model.eval()
    tokens, offset = encoded[plan["schedule"][0]["record_index"]]
    logits = model(mx.array([tokens[:offset]]))[:, -1, :].astype(mx.float32)
    mx.eval(logits)
    mx.save_safetensors(str(output / "reload_reference.safetensors"), {"logits": logits})
    save_json(output / "training_result.json", {"status": "completed", "optimizer_updates": int(optimizer.step.item()),
              "trainable_parameters": trainable_count, "adapter_digest_before": before, "adapter_digest_after": after,
              "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3, "reference_prompt_tokens": tokens[:offset],
              "adapter_sha256": sha256(output / "adapter/adapters.safetensors"), "base_weights_written": False,
              "checkpoint_kind": "adapter weights only; optimizer state is not saved for resume"})


def evaluate_worker(output):
    plan, config, probe = start_worker(output)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters
    trained = json.loads((output / "training_result.json").read_text())
    if trained["status"] != "completed" or sha256(output / "adapter/adapters.safetensors") != trained["adapter_sha256"]:
        raise ValueError("Require a completed, unchanged training checkpoint.")
    model, tokenizer = load(str(HERE / config["model"]), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    load_adapters(model, str(output / "adapter"))
    arrays = [(name, value) for name, value in tree_flatten(model.parameters()) if name.endswith((".lora_a", ".lora_b"))]
    if adapter_digest(arrays) != trained["adapter_digest_after"]:
        raise ValueError("Reloaded adapter numbers differ from the trained numbers.")
    model.eval()
    reference = mx.load(str(output / "reload_reference.safetensors"))["logits"]
    actual = model(mx.array([trained["reference_prompt_tokens"]]))[:, -1, :].astype(mx.float32)
    difference = float(mx.max(mx.abs(reference - actual)).item())
    agrees = bool(mx.allclose(reference, actual, atol=0.00001, rtol=0.00001).item())
    save_json(output / "reload_check.json", {"adapter_numbers_match": True, "logits_match": agrees,
              "max_absolute_logit_difference": difference, "atol": 0.00001, "rtol": 0.00001, "fresh_process": True})
    if not agrees:
        raise ValueError("Reloaded model output differs from the pre-save model output.")
    del arrays, reference, actual
    mx.clear_cache()
    sample = read_rows(ROOT / SNAPSHOT / "sample.jsonl")
    baseline_plan = json.loads((ROOT / SNAPSHOT / "plan.json").read_text())
    limits = baseline_plan["limits"]
    results = []
    with (output / "evaluation_predictions.jsonl").open("x") as stream:
        for index, row in enumerate(sample):
            for swapped in ((False, True) if index % 2 == 0 else (True, False)):
                started = time.perf_counter()
                result = {"id": row["id"], "swapped": swapped, "target": row["target"], "decision": None}
                try:
                    prompt = tokenizer.apply_chat_template(messages_for(row, swapped, baseline_plan["rubric"]), tokenize=True, add_generation_prompt=True)
                    result["input_tokens"] = len(prompt)
                    if len(prompt) > limits["input_tokens"]:
                        result["status"] = "input_too_long"
                    else:
                        raw, finish, output_tokens = "", None, 0
                        for piece in stream_generate(model, tokenizer, prompt=prompt, max_tokens=limits["output_tokens"], sampler=make_sampler(temp=0.0)):
                            raw += piece.text
                            finish, output_tokens = piece.finish_reason, piece.generation_tokens
                        result.update(raw_output=raw, finish_reason=finish, output_tokens=output_tokens)
                        result["status"], result["decision"] = classify_output(raw, finish, swapped)
                except Exception as error:
                    result.update(status="error", error_type=type(error).__name__)
                result.update(seconds=time.perf_counter() - started, mlx_peak_memory_gib=mx.get_peak_memory() / 1024**3)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                results.append(result)
                print(f"Evaluate {len(results)}/120: {result['status']}", flush=True)
                mx.clear_cache()
    save_json(output / "evaluation_runtime.json", {"mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3})


def supervise(output, phase, seconds):
    result = {"phase": phase, "status": "running", "wall_limit_seconds": seconds}
    save_json(output / (phase + "_worker.json"), result)
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--" + phase, str(output)], timeout=seconds, check=False)
        result["status"] = "completed" if process.returncode == 0 else "worker_failed"
    except subprocess.TimeoutExpired:
        result["status"] = "deadline_exceeded"
    result["wall_seconds"] = time.perf_counter() - started
    save_json(output / (phase + "_worker.json"), result)
    return result


def collect_predictions(path):
    if not path.exists():
        return [], False
    records, damaged = [], False
    with path.open() as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                damaged = True
                break
    return records, damaged


def main():
    guard = verify()
    config, probe = check_ready(READY)
    index = json.loads((READY / "trial_index.json").read_text())
    sample = read_rows(ROOT / SNAPSHOT / "sample.jsonl")
    if {row["group_id"] for row in index} & {row["group_id"] for row in sample}:
        raise ValueError("Training groups overlap the protected validation baseline.")
    output = HERE / "results" / ("training_trial_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    plan = {"kind": "first_20_update_lora_trial", "config": config, "prepared_folder": str(READY.relative_to(ROOT)),
            "schedule": schedule(index, config["future_trial_iterations"], config["seed"]), "baseline_guard_before": guard,
            "code_sha256": {name: sha256(HERE / name) for name in ("train_trial.py", "prepare_training.py", "probe_training.py", "baseline_guard.py")},
            "training_profile_sha256": sha256(READY / "profile.json"), "evaluation_wall_seconds": 900,
            "baseline_sample_sha256": sha256(ROOT / SNAPSHOT / "sample.jsonl"), "automatic_retries": 0,
            "paid_api_calls": 0, "calibration_or_test_opened": False, "selection": "fixed final update; no checkpoint chosen using validation scores"}
    save_json(output / "plan.json", plan)
    print("Trial output:", output, flush=True)
    training = supervise(output, "train", config["future_trial_wall_seconds"])
    evaluation = {"status": "not_run"}
    if training["status"] == "completed":
        evaluation = supervise(output, "evaluate", plan["evaluation_wall_seconds"])
    records, damaged = collect_predictions(output / "evaluation_predictions.jsonl")
    baseline = json.loads((ROOT / SNAPSHOT / "summary.json").read_text())["metrics"]
    after = summarize(sample, records)
    report = {"training": training, "evaluation": evaluation, "damaged_prediction_tail": damaged,
              "baseline": baseline, "trial": after, "baseline_guard_after": verify(),
              "interpretation": "Small development preference comparison, not final-test accuracy or evidence-grounding reliability."}
    save_json(output / "comparison.json", report)
    print("Saved comparison:", output / "comparison.json", flush=True)
    if training["status"] != "completed" or evaluation["status"] != "completed" or damaged:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 3 and sys.argv[1] in ("--train", "--evaluate"):
        {"--train": train_worker, "--evaluate": evaluate_worker}[sys.argv[1]](Path(sys.argv[2]).resolve())
    else:
        raise SystemExit("Run train_trial.py without arguments.")
