"""Prepare corrected binary training with a training-side selection holdout."""
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from baseline import read_rows, save_json
from baseline_guard import ROOT, sha256
from compare_tie_prompt import history_guard, verify_receipt
from judge_v2 import DATA, HERE, TOKENS, messages

SEED = 20261007
SOURCES = ("prepare_judge.py", "train_judge.py", "test_train_judge.py", "judge_head.py")


def history():
    result = history_guard()
    result["prompt_trial"] = verify_receipt(HERE / "results/tie_prompt_20261006T230223795161Z", "prompt_verification.json")
    lock = json.loads((HERE / "baseline_v2.lock.json").read_text())
    folder = ROOT / lock["run_folder"]
    if sha256(folder / lock["verification_receipt"]) != lock["verification_receipt_sha256"]:
        raise ValueError("The baseline-v2 reference receipt changed.")
    result["baseline_v2"] = verify_receipt(folder, lock["verification_receipt"])
    result["baseline_v2_lock_sha256"] = sha256(HERE / "baseline_v2.lock.json")
    return result


def select_groups(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    chosen = []
    for key in keys:
        group = sorted(groups[key], key=lambda row: row["id"])
        if len(chosen) + len(group) <= count:
            chosen.extend(group)
    if len(chosen) != count:
        raise ValueError("Cannot select the requested count without splitting a group.")
    return chosen


def encode(row, tokenizer, rubric):
    label = row["target"]["preference"]
    if label not in ("A", "B"):
        raise ValueError("Only clear A/B preferences can become training targets.")
    item = {"id": row["id"], "group_id": row["group_id"], "label": label,
            "strength": abs(row["target"]["raw_preference_strength"]), "orders": []}
    for swapped in (False, True):
        prompt = messages(row, swapped, "json", rubric)
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        target = int(label == "B") ^ int(swapped)
        answer = "AB"[target]
        full = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=True, return_dict=False)
        if full[:len(ids)] != ids or full[len(ids)] != TOKENS[answer]:
            raise ValueError("The answer does not begin at the verified decision position.")
        item["orders"].append({"swapped": swapped, "tokens": ids, "target": target})
    return item


def write_rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare():
    before = history()
    environment = {}
    for line in (HERE / "requirements.lock.txt").read_text().splitlines():
        package, expected = line.split("==")
        environment[package] = version(package)
        if environment[package] != expected:
            raise ValueError("Pinned dependency changed: " + package)
    release = json.loads((DATA / "report.json").read_text())
    for split in ("train", "validation"):
        if sha256(DATA / (split + ".jsonl")) != release["output_sha256"][split + ".jsonl"]:
            raise ValueError("Prepared source data changed.")
    train, validation = (read_rows(DATA / (split + ".jsonl")) for split in ("train", "validation"))
    if {r["group_id"] for r in train} & {r["group_id"] for r in validation}:
        raise ValueError("Training and validation groups overlap.")
    lock = json.loads((HERE / "baseline_v2.lock.json").read_text())
    reference = json.loads((ROOT / lock["run_folder"] / "plan.json").read_text())
    rubric = reference["rubric"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HERE / "models/Qwen3-4B-Instruct-2507", local_files_only=True, trust_remote_code=False)
    if [tokenizer.encode(x, add_special_tokens=False) for x in "AB"] != [[32], [33]]:
        raise ValueError("The A/B token identities changed.")
    encoded = [encode(row, tokenizer, rubric) for row in train if row["target"]["preference"] in ("A", "B")]
    too_long = {r["group_id"] for r in encoded if max(len(o["tokens"]) for o in r["orders"]) > 2048}
    eligible = [r for r in encoded if r["group_id"] not in too_long]
    selected = select_groups(eligible, 2000, SEED)
    pool_groups = {r["group_id"] for r in selected}
    monitor = select_groups([r for r in eligible if r["group_id"] not in pool_groups], 200, SEED + 1)
    monitor_groups = {r["group_id"] for r in monitor}
    if pool_groups & monitor_groups or monitor_groups & {r["group_id"] for r in validation}:
        raise ValueError("Selection-monitor groups overlap training or validation.")
    if max(len(o["tokens"]) for r in monitor for o in r["orders"]) > 4096:
        raise ValueError("Validation exceeds its full-length limit.")
    ordered = sorted(selected, key=lambda r: (max(len(o["tokens"]) for o in r["orders"]), r["id"]))
    probes = [ordered[int((i + .5) * len(ordered) / 8)] for i in range(8)]
    if len({r["group_id"] for r in probes}) != 8:
        raise ValueError("Probe quantiles must use distinct question groups.")
    longest = max(eligible, key=lambda r: max(len(o["tokens"]) for o in r["orders"]))
    output = HERE / "results" / ("binary_readiness_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    (output / "source_snapshot").mkdir()
    for name in SOURCES:
        shutil.copyfile(HERE / name, output / "source_snapshot" / name)
    for name, rows in (("train_pool", selected), ("selection_monitor", monitor), ("probe_pairs", probes), ("longest_pair", [longest])):
        write_rows(output / (name + ".jsonl"), rows)
    profile = {"clear_training_pairs": len(encoded), "eligible_pairs": len(eligible), "excluded_pairs_by_group_length": len(encoded) - len(eligible),
               "excluded_groups": len(too_long), "selected_pairs": len(selected), "selected_groups": len({r["group_id"] for r in selected}),
               "monitor_pairs": len(monitor), "monitor_groups": len({r["group_id"] for r in monitor}),
               "monitor_source": "eligible training groups outside the 2000-pair pool; excluded from all later gradient training",
               "selected_strengths": dict(Counter(r["strength"] for r in selected)),
               "mean_selected_prompt_tokens": sum(len(o["tokens"]) for r in selected for o in r["orders"]) / 4000,
               "probe_lengths": [[len(o["tokens"]) for o in r["orders"]] for r in probes],
               "longest_lengths": [len(o["tokens"]) for o in longest["orders"]]}
    save_json(output / "profile.json", profile)
    config = {"model": "models/Qwen3-4B-Instruct-2507", "num_layers": 36,
              "lora_parameters": {"rank": 16, "scale": 20.0, "dropout": .05}, "optimizer": "Adam",
              "learning_rate": 2e-5, "warmup_updates": 50, "schedule_updates": 150, "final_lr_fraction": .1,
              "examples_per_update": 8, "microbatch_size": 1, "training_token_cap": 2048,
              "adam_bias_correction": True, "gradient_clip_norm": 1.0,
              "dropout_randomness": "explicit per-presentation keys, saved private PRNG state; masks reused during gradient recomputation",
              "training_output_head": "cast final hidden state and selected A/B embedding rows before float32 projection",
                   "gradient_evaluation": "materialize loss and gradients before computing their diagnostic norm",
              "gradient_checkpointing": True, "memory_limit_gib": 18, "seed": SEED}
    plan = {"status": "planned_before_gpu_work", "kind": "binary_training_readiness_not_quality_experiment",
            "hypothesis": "All-block rank-16 binary LoRA fits the memory guideline and resumes exactly with paired accumulation.",
            "format": "baseline_v2_json", "rubric": rubric, "config": config, "history_before": before,
            "environment": environment, "source_sha256": {name: sha256(HERE / name) for name in SOURCES},
            "reused_source_sha256": {name: sha256(HERE / name) for name in ("baseline.py", "baseline_guard.py", "compare_tie_prompt.py", "judge_v2.py", "train_trial.py")},
            "files_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            "source_data_sha256": {name: sha256(DATA / (name + ".jsonl")) for name in ("train", "validation")},
            "phase_wall_seconds": {"checks": 120, "continuous": 1200, "resumed": 600},
            "updates": {"continuous": 2, "resumed": 1}, "long_sequence_gradient_presentations": 2,
            "run_long_stress": True,
            "acceptance": {"memory_gib_at_most": 18, "finite_nonzero_gradients": True, "only_adapter_gradients": True,
                           "resume": "exact adapter, optimizer and random-state digests after update 2; identical loss",
                           "loss_absolute_tolerance": 1e-6, "scoring_probability_tolerance": .01},
            "automatic_retries": 0, "paid_calls": 0, "calibration_or_test_opened": False,
            "future_sweep": {"status": "planned_not_run", "learning_rates": [1e-5, 2e-5, 5e-5], "updates_each": 150,
                             "examples_each": 1200, "monitor_every_updates": 100, "monitor_at_final_update": True,
                             "choose_by": "lowest fixed-200-pair training-side monitor log-loss; full validation after selection",
                             "budget": "Set from the readiness measurement before launching."}}
    save_json(output / "plan.json", plan)
    print(json.dumps({"folder": str(output), "profile": profile}, indent=2))


if __name__ == "__main__":
    prepare()
