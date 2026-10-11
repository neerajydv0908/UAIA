"""Expand the prepared training pool to 200 updates while preserving earlier runs."""
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from baseline import read_rows, save_json, summarize
from baseline_guard import ROOT, SNAPSHOT, sha256, verify
from train_trial import HERE, READY, adapter_digest, check_ready, collect_predictions, evaluate_worker, train_worker

PREVIOUS = HERE / "results/training_trial_20261006T104025828291Z"
TRAIN_SECONDS = 1200
EVALUATION_SECONDS = 900


def expanded_schedule(index, earlier, seed):
    if len(index) != 100 or len({row["group_id"] for row in index}) != 100:
        raise ValueError("Require exactly 100 distinct prepared training groups.")
    if len(earlier) != 20:
        raise ValueError("Require the exact 20-update starting schedule.")
    used = set()
    for first, second in zip(earlier[::2], earlier[1::2]):
        positions = []
        for item in (first, second):
            position, orientation = divmod(item["record_index"], 2)
            if not 0 <= position < len(index) or orientation != int(item["swapped"]):
                raise ValueError("A prior record index or orientation is invalid.")
            row = index[position]
            if (item["id"], item["group_id"]) != (row["id"], row["group_id"]):
                raise ValueError("Prior schedule no longer matches the prepared index.")
            positions.append(position)
        if positions[0] != positions[1] or positions[0] in used or first["swapped"] == second["swapped"]:
            raise ValueError("Earlier updates must contain ten unique paired groups.")
        used.add(positions[0])
    remaining = [position for position in range(len(index)) if position not in used]
    random.Random(seed).shuffle(remaining)
    result = [dict(item) for item in earlier]
    for order, position in enumerate(remaining, len(used)):
        for swapped in ((False, True) if order % 2 == 0 else (True, False)):
            result.append({"record_index": 2 * position + int(swapped), "id": index[position]["id"],
                           "group_id": index[position]["group_id"], "swapped": swapped})
    return result


def verify_previous(folder=PREVIOUS):
    receipt = json.loads((folder / "trial_verification.json").read_text())
    if receipt["status"] != "passed":
        raise ValueError("The previous trial must have a passed artifact verification.")
    for name, expected in receipt["artifact_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError("Earlier trial artifact changed: " + name)
    return {"status": "passed", "files_checked": len(receipt["artifact_sha256"]),
            "receipt_sha256": sha256(folder / "trial_verification.json")}


def paired_changes(before, after, swapped):
    old = {(row["id"], row["swapped"]): row for row in before}
    result = {"improved": 0, "regressed": 0, "both_match": 0, "both_mismatch": 0, "unresolved": 0}
    current = {(row["id"], row["swapped"]): row for row in after}
    for key, previous in old.items():
        if key[1] != swapped:
            continue
        row = current.get(key)
        if row is None or row["status"] != "completed" or previous["status"] != "completed":
            result["unresolved"] += 1
            continue
        if row["target"] != previous["target"]:
            raise ValueError("Compared records disagree on the expected label.")
        was = previous["decision"] == previous["target"]
        now = row["decision"] == row["target"]
        result[{(False, True): "improved", (True, False): "regressed", (True, True): "both_match", (False, False): "both_mismatch"}[(was, now)]] += 1
    return result


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


def train_and_check_prefix(output):
    verify_previous()
    train_worker(output)
    import mlx.core as mx
    earlier = json.loads((PREVIOUS / "training_result.json").read_text())
    arrays = mx.load(str(output / "checkpoint_0020/adapters.safetensors"))
    actual = adapter_digest(list(arrays.items()))
    matched = actual == earlier["adapter_digest_after"]
    save_json(output / "prefix_check.json", {"status": "passed" if matched else "failed", "updates": 20,
              "adapter_numeric_fingerprint_matches_previous_trial": matched,
              "previous_digest": earlier["adapter_digest_after"], "expanded_step20_digest": actual})
    if not matched:
        raise ValueError("The first 20 updates did not reproduce the earlier adapter; evaluation stopped.")


def main():
    baseline_guard = verify()
    earlier_guard = verify_previous()
    config, probe = check_ready(READY)
    index = json.loads((READY / "trial_index.json").read_text())
    previous_plan = json.loads((PREVIOUS / "plan.json").read_text())
    if config != previous_plan["config"]:
        raise ValueError("Keep the earlier model, optimizer, and readiness settings fixed.")
    sample = read_rows(ROOT / SNAPSHOT / "sample.jsonl")
    if {row["group_id"] for row in index} & {row["group_id"] for row in sample}:
        raise ValueError("Training groups overlap the baseline validation sample.")
    output = HERE / "results" / ("expanded_trial_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    plan = {"kind": "expanded_200_update_lora_trial", "config": config, "prepared_folder": str(READY.relative_to(ROOT)),
            "schedule": expanded_schedule(index, previous_plan["schedule"], config["seed"]),
            "planned_updates": 200, "training_wall_seconds": TRAIN_SECONDS, "evaluation_wall_seconds": EVALUATION_SECONDS,
            "baseline_guard_before": baseline_guard, "previous_trial_guard_before": earlier_guard,
            "code_sha256": {name: sha256(HERE / name) for name in ("expand_trial.py", "train_trial.py", "prepare_training.py", "probe_training.py", "baseline_guard.py")},
            "training_profile_sha256": sha256(READY / "profile.json"), "baseline_sample_sha256": sha256(ROOT / SNAPSHOT / "sample.jsonl"),
            "previous_plan_sha256": sha256(PREVIOUS / "plan.json"), "automatic_retries": 0,
            "paid_api_calls": 0, "calibration_or_test_opened": False,
            "selection": "fixed final update 200; no intermediate checkpoint chosen from validation results",
            "restart_policy": "Fresh original model and optimizer; exact 20-update prefix followed by 180 additional presentations; no automatic resume",
            "config_note": "Readiness config retains its historical 20-update proposal; the actual schedule and wall budgets above govern this run."}
    save_json(output / "plan.json", plan)
    print("Expanded trial output:", output, flush=True)
    training = supervise(output, "train", TRAIN_SECONDS)
    evaluation = {"status": "not_run"}
    if training["status"] == "completed":
        evaluation = supervise(output, "evaluate", EVALUATION_SECONDS)
    records, damaged = collect_predictions(output / "evaluation_predictions.jsonl")
    baseline = json.loads((ROOT / SNAPSHOT / "summary.json").read_text())["metrics"]
    first = json.loads((PREVIOUS / "comparison.json").read_text())["trial"]
    after = summarize(sample, records)
    previous_predictions = read_rows(PREVIOUS / "evaluation_predictions.jsonl")
    baseline_predictions = read_rows(ROOT / SNAPSHOT / "predictions.jsonl")
    report = {"training": training, "evaluation": evaluation, "damaged_prediction_tail": damaged,
              "baseline": baseline, "first_20_update_trial": first, "expanded_trial": after,
              "paired_changes": {name: {str(swapped): paired_changes(before, records, swapped) for swapped in (False, True)}
                                 for name, before in (("from_baseline", baseline_predictions), ("from_20_updates", previous_predictions))},
              "baseline_guard_after": verify(), "previous_trial_guard_after": verify_previous(),
              "interpretation": "Repeated development preference comparison, not final-test accuracy or evidence-grounding reliability."}
    save_json(output / "comparison.json", report)
    print("Saved expanded comparison:", output / "comparison.json", flush=True)
    if training["status"] != "completed" or evaluation["status"] != "completed" or damaged:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 3 and sys.argv[1] in ("--train", "--evaluate"):
        {"--train": train_and_check_prefix, "--evaluate": evaluate_worker}[sys.argv[1]](Path(sys.argv[2]).resolve())
    else:
        raise SystemExit("Run expand_trial.py without arguments.")
