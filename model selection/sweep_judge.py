"""Preregister, check, run, and report the bounded binary learning-rate sweep."""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from baseline import read_rows, save_json
from baseline_guard import sha256
from compare_tie_prompt import verify_receipt
from judge_v2 import DATA, HERE, digest_text, first_logits, messages, probabilities
from judge_v2_metrics import combine, compare, holm, measure, summary
from prepare_judge import history, write_rows
from train_judge import load_training, restore_checkpoint, save_checkpoint, schedule_pairs, state_digests, update

READY = HERE / "results/training_memory_20261007T090838442782Z"
BASELINE = HERE / "results/baseline_v2_20261007T004058039152Z"
RATES = {"lr_1e-5": 1e-5, "lr_2e-5": 2e-5, "lr_5e-5": 5e-5}
STEPS = (25, 50, 75, 100, 125, 150)
MONITORS = (100, 150)
SOURCES = ("sweep_judge.py", "test_sweep_judge.py")
REUSED = ("train_judge.py", "prepare_judge.py", "test_train_judge.py", "judge_v2.py", "judge_v2_metrics.py", "baseline.py", "baseline_guard.py", "compare_tie_prompt.py", "train_trial.py")


def read_json(path):
    return json.loads(path.read_text())


def status(path, value):
    temporary = path.with_suffix(".tmp")
    save_json(temporary, value)
    temporary.replace(path)


def historical_checks():
    result = history()
    result["binary_readiness"] = verify_receipt(HERE / "results/binary_readiness_20261007T032609258145Z", "readiness_verification.json")
    result["training_memory"] = verify_receipt(READY, "memory_verification.json")
    return result


def jobs():
    return {f"{name}_{end:04d}": {"candidate": name, "start": end - 25, "end": end,
                                "wall_seconds": 4200 if end in MONITORS else 3000}
            for name in RATES for end in STEPS}


def prepare():
    before = historical_checks()
    ready, baseline = read_json(READY / "plan.json"), read_json(BASELINE / "plan.json")
    if read_json(READY / "measurement_summary.json")["status"] != "full_length_readiness_passed":
        raise ValueError("Full-length readiness must pass before planning the sweep.")
    for name, expected in ready["environment"].items():
        if version(name) != expected:
            raise ValueError("Installed dependency changed: " + name)
    if shutil.disk_usage(HERE).free < 12 * 1024**3:
        raise ValueError("Reserve at least 12 GiB for complete sweep checkpoints.")
    output = HERE / "results" / ("judge_sweep_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    for name in ("train_pool.jsonl", "validation_monitor.jsonl", "probe_pairs.jsonl"):
        shutil.copy2(READY / name, output / name)
    shutil.copy2(BASELINE / "sample.jsonl", output / "validation_sample.jsonl")
    shutil.copy2(BASELINE / "baseline_v2_json/paired_scores.json", output / "baseline_pairs.json")
    sample = read_rows(output / "validation_sample.jsonl")
    expected = {(r["id"], r["swapped"]): r for r in read_json(BASELINE / "input_checks.json")["lengths"] if r["style"] == "json"}
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HERE / ready["config"]["model"], local_files_only=True, trust_remote_code=False)
    encoded = []
    for row in sample:
        orders = []
        for swapped in (False, True):
            prompt = messages(row, swapped, "json", ready["rubric"])
            tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
            original = expected[row["id"], swapped]
            if len(tokens) != original["tokens"] or len(tokens) > 4096 or digest_text(json.dumps(prompt, ensure_ascii=False)) != original["messages_sha256"]:
                raise ValueError("Validation input differs from baseline v2.")
            orders.append({"swapped": swapped, "tokens": tokens})
        encoded.append({"id": row["id"], "orders": orders})
    write_rows(output / "validation_inputs.jsonl", encoded)
    by_id = {r["id"]: r for r in encoded}
    monitor = read_rows(output / "validation_monitor.jsonl")
    for row in monitor:
        if [o["tokens"] for o in row["orders"]] != [o["tokens"] for o in by_id[row["id"]]["orders"]]:
            raise ValueError("The prepared monitor changed its input format.")
    pool = read_rows(output / "train_pool.jsonl")
    if len(pool) != 2000 or len(monitor) != 100 or {r["group_id"] for r in pool} & {r["group_id"] for r in sample}:
        raise ValueError("Unexpected pool size or train/validation group overlap.")
    settings = baseline["statistics"]
    plan = {"status": "preregistered_before_sweep", "hypothesis": "Broader binary LoRA improves probability quality; select the learning rate by fixed-monitor log-loss.",
            "config": ready["config"], "rubric": ready["rubric"], "rates": RATES, "jobs": jobs(), "updates_each": 150,
            "examples_each": 1200, "unique_training_pairs_each": 600, "data_order": "First 600 pairs in the unchanged seeded 2000-pair pool, both orders in each update.",
            "monitor_updates": list(MONITORS), "checkpoint_updates": list(STEPS), "statistics": settings,
            "selection": "Minimum monitor log-loss among all six complete checkpoints; exact ties prefer lower learning rate, then earlier update. No selection from incomplete runs.",
            "development_screen": "Accuracy gain >= .03, paired CI lower > 0, exact McNemar p < .05 and no log-loss rise; OR log-loss reduction >= .02, paired CI upper < 0 and no observed accuracy decline. Selection uses the same validation data: exploratory development evidence, not an independent confirmation.",
            "stopping": "Exactly 150 updates per rate; no performance early stopping in this small sweep. Stop the workflow on invalid numbers, memory violation, worker failure, deadline, or changed inputs; never retry automatically.",
            "budgets": {"candidate_seconds": 14400, "sweep_and_full_seconds": 46800, "full_validation_seconds": 3600, "probe_seconds_each": 900},
            "timing_estimate": "About 7.4 hours for 3600 training presentations, plus 1200 monitor and 792 full-validation score reads; roughly 8–9 hours overall, with machine-dependent overhead.",
            "history_before": before, "environment": ready["environment"], "source_data_sha256": ready["source_data_sha256"],
            "source_sha256": {name: sha256(HERE / name) for name in SOURCES}, "reused_source_sha256": {name: sha256(HERE / name) for name in REUSED},
            "files_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            "automatic_retries": 0, "paid_calls": 0, "calibration_or_test_opened": False}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in SOURCES + REUSED:
        shutil.copy2(HERE / name, snapshot / name)
    save_json(output / "plan.json", plan)
    print(output, flush=True)


def validate(output):
    plan = read_json(output / "plan.json")
    for key in ("source_sha256", "reused_source_sha256"):
        for name, expected in plan[key].items():
            if sha256(HERE / name) != expected:
                raise ValueError("Source changed after preregistration: " + name)
    for name, expected in plan["files_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError("Prepared input changed: " + name)
    return plan


def evaluate(model, optimizer, encoded, sample, destination, plan):
    import mlx.core as mx
    destination.mkdir(exist_ok=False)
    before, was_training = state_digests(model, optimizer), model.training
    model.eval()
    scores = []
    try:
        with (destination / "scores.jsonl").open("x") as stream:
            for index, row in enumerate(encoded):
                for order in row["orders"]:
                    if not 0 < len(order["tokens"]) <= 4096:
                        raise ValueError("Validation input must fit without truncation.")
                    tick = time.perf_counter()
                    logits = first_logits(model, order["tokens"], 512)
                    selected = logits[mx.array([32, 33, 48731])].tolist()
                    p, three = probabilities(selected)
                    if mx.get_peak_memory() > plan["config"]["memory_limit_gib"] * 1024**3:
                        raise ValueError("Validation exceeded the memory guideline.")
                    entry = {"id": row["id"], "swapped": order["swapped"], "status": "completed", "logits": selected,
                             "p_display_a": p, "three_way_prefix_probs": three, "input_tokens": len(order["tokens"]), "seconds": time.perf_counter() - tick}
                    scores.append(entry)
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                    del logits
                    mx.clear_cache()
                status(destination / "progress.json", {"pairs": index + 1, "planned": len(encoded)})
        paired = combine(sample, scores)
        measured = measure([r for r in paired if r["target"] in ("A", "B")], "averaged", plan["statistics"])
        if measured["status"] != "completed" or state_digests(model, optimizer) != before:
            raise ValueError("Evaluation is incomplete or changed training state.")
        save_json(destination / "paired_scores.json", paired)
        save_json(destination / "metrics.json", measured)
        save_json(destination / "state_check.json", {"status": "passed", "before": before, "after": state_digests(model, optimizer)})
        return measured
    finally:
        model.train(was_training)


def monitor_inputs(output, limit=None):
    encoded = read_rows(output / "validation_monitor.jsonl")[:limit]
    sample = {r["id"]: r for r in read_rows(output / "validation_sample.jsonl")}
    return encoded, [sample[r["id"]] for r in encoded]


def worker(output, job):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    plan = validate(output)
    import mlx.core as mx
    import mlx.nn as nn
    from train_judge import binary_loss
    destination = output / job
    destination.mkdir(exist_ok=False)
    if job == "checks":
        from test_train_judge import gpu_checks
        save_json(destination / "result.json", gpu_checks(plan["config"]))
        return
    config = dict(plan["config"])
    spec = plan["jobs"].get(job)
    if spec:
        config["learning_rate"] = plan["rates"][spec["candidate"]]
    if job == "full":
        selection = read_json(output / "selection.json")
        config["learning_rate"] = selection["learning_rate"]
    model, optimizer, architecture = load_training(config)
    save_json(destination / "architecture.json", architecture)
    if job == "full":
        restore_checkpoint(output / selection["checkpoint"], model, optimizer, config, output)
        evaluate(model, optimizer, read_rows(output / "validation_inputs.jsonl"), read_rows(output / "validation_sample.jsonl"), destination / "evaluation", plan)
        save_json(destination / "result.json", {"status": "passed", "optimizer_updates": 0})
        return
    probe = job in ("probe_train", "probe_resume")
    if not probe and spec is None:
        raise ValueError("Unknown worker job.")
    pairs = read_rows(output / ("probe_pairs.jsonl" if probe else "train_pool.jsonl"))
    batches = schedule_pairs(pairs)
    start, end = (0, 2) if probe else (spec["start"], spec["end"])
    if job == "probe_resume":
        start = restore_checkpoint(output / "probe_train/checkpoint_0001", model, optimizer, config, output)
    elif start:
        previous = f"{spec['candidate']}_{start:04d}/checkpoint_{start:04d}"
        if restore_checkpoint(output / previous, model, optimizer, config, output) != start:
            raise ValueError("The restored update does not match this segment.")
    initial = state_digests(model, optimizer)
    calculate = nn.value_and_grad(model, binary_loss)
    encoded, sample = monitor_inputs(output, 2 if probe else None)
    if job == "probe_resume":
        evaluate(model, optimizer, encoded, sample, destination / "monitor_0001", plan)
    with (destination / "updates.jsonl").open("x") as stream:
        for batch in batches[start:end]:
            entry = update(model, optimizer, calculate, batch, config)
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            status(destination / "progress.json", {"step": entry["step"], "end": end, "mean_loss": entry["mean_loss"], "memory_gib": entry["mlx_peak_memory_gib"]})
            print(json.dumps({"job": job, "step": entry["step"], "loss": entry["mean_loss"]}), flush=True)
            if probe or entry["step"] == end:
                save_checkpoint(destination / f"checkpoint_{entry['step']:04d}", model, optimizer, config, output)
            if probe or entry["step"] in MONITORS:
                evaluate(model, optimizer, encoded, sample, destination / f"monitor_{entry['step']:04d}", plan)
    save_json(destination / "result.json", {"status": "passed", "initial": initial, "final": state_digests(model, optimizer),
              "start": start, "end": end, "updates_applied": end - start, "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3})


def rank_checkpoints(records):
    expected = {(name, step) for name in RATES for step in MONITORS}
    if len(records) != len(expected) or {(r["candidate"], r["step"]) for r in records} != expected:
        raise ValueError("Require all six distinct monitor results before selecting.")
    for row in records:
        if row["completed"] != 100 or row["groups"] != 100 or not math.isfinite(row["log_loss"]) or row["log_loss"] < 0:
            raise ValueError("Require complete finite monitor measurements.")
    return min(records, key=lambda r: (r["log_loss"], RATES[r["candidate"]], r["step"]))


def select(output):
    records = []
    first_states = []
    for candidate in RATES:
        previous = None
        for step in STEPS:
            result = read_json(output / f"{candidate}_{step:04d}/result.json")
            if result["status"] != "passed" or result["start"] != step - 25 or result["end"] != step or result["updates_applied"] != 25:
                raise ValueError("Every candidate must complete all 150 updates.")
            if previous is not None and previous != result["initial"]:
                raise ValueError("A segment did not resume its predecessor's numerical state.")
            if step == 25:
                first_states.append({key: result["initial"][key] for key in ("adapter", "random")})
            previous = result["final"]
    if any(state != first_states[0] for state in first_states):
        raise ValueError("Candidates did not start from identical adapters and random state.")
    for candidate in RATES:
        for step in MONITORS:
            job = f"{candidate}_{step:04d}"
            if read_json(output / job / "result.json")["status"] != "passed":
                raise ValueError("A candidate has not completed.")
            measured = read_json(output / job / f"monitor_{step:04d}/metrics.json")
            if measured["status"] != "completed":
                raise ValueError("A monitor has not completed.")
            records.append({"candidate": candidate, "step": step, "completed": measured["completed"], "groups": measured["groups"],
                            "log_loss": measured["metrics"]["log_loss"]["value"], "checkpoint": f"{job}/checkpoint_{step:04d}"})
    chosen = rank_checkpoints(records)
    save_json(output / "selection.json", {**chosen, "learning_rate": RATES[chosen["candidate"]], "all_candidates": records,
              "status": "selected_for_full_validation_not_promoted", "selected_before_full_validation": True})


def check_probe(output):
    left, right = (read_json(output / job / "result.json") for job in ("probe_train", "probe_resume"))
    if left["final"] != right["final"] or left["end"] != right["end"] or right["start"] != 1:
        raise ValueError("Interleaved evaluation changed exact checkpoint resumption.")
    a, b = read_rows(output / "probe_train/updates.jsonl")[1], read_rows(output / "probe_resume/updates.jsonl")[0]
    if a["mean_loss"] != b["mean_loss"] or [{k: v for k, v in r.items() if k != "seconds"} for r in a["presentations"]] != [{k: v for k, v in r.items() if k != "seconds"} for r in b["presentations"]]:
        raise ValueError("The replay changed second-update losses or gradients.")
    for step in (1, 2):
        a, b = (read_json(output / job / f"monitor_{step:04d}/paired_scores.json") for job in ("probe_train", "probe_resume"))
        if a != b:
            raise ValueError("Resumed monitor probabilities differ.")
    save_json(output / "probe_verification.json", {"status": "passed", "interleaved_monitor_preserves_state": True,
              "fresh_process_resume_exact": True, "monitor_probabilities_exact": True, "probe_optimizer_applications": 3, "probe_score_reads": 16})


def launch(output, job, limit):
    marker = output / (job + "_worker.json")
    if marker.exists():
        if read_json(marker)["status"] == "completed" and read_json(output / job / "result.json")["status"] == "passed":
            return
        raise ValueError("Preserve the unsuccessful attempt; automatic retries are forbidden: " + job)
    started = time.perf_counter()
    status(marker, {"status": "running", "wall_limit_seconds": limit})
    with (output / (job + ".log")).open("x") as stream:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", str(output), job], stdout=stream, stderr=subprocess.STDOUT, timeout=limit, check=False)
            outcome = "completed" if process.returncode == 0 else "worker_failed"
        except subprocess.TimeoutExpired:
            outcome = "deadline_exceeded"
    status(marker, {"status": outcome, "seconds": time.perf_counter() - started, "wall_limit_seconds": limit})
    print(json.dumps({"job": job, "status": outcome, "seconds": time.perf_counter() - started}), flush=True)
    if outcome != "completed":
        raise RuntimeError("Stopped at " + job + "; inspect its preserved log and failure record.")


def report(output):
    plan, selected = validate(output), read_json(output / "selection.json")
    after, before = read_json(output / "full/evaluation/paired_scores.json"), read_json(output / "baseline_pairs.json")
    paired = compare(before, after, plan["statistics"])
    holm({"selected_vs_baseline": paired})
    accuracy, loss = (paired["delta_after_minus_before"][key] for key in ("accuracy", "log_loss"))
    settings = plan["statistics"]
    accuracy_pass = accuracy["value"] >= settings["minimum_accuracy_gain"] and accuracy["ci95"][0] > 0 and paired["mcnemar"]["exact_two_sided_p"] < settings["familywise_alpha"] and loss["value"] <= 0
    probability_pass = loss["value"] <= -settings["minimum_logloss_reduction"] and loss["ci95"][1] < 0 and accuracy["value"] >= 0
    measured = summary(after, plan["statistics"])
    history_after = historical_checks()
    if history_after != plan["history_before"]:
        raise ValueError("A historical reference changed during the sweep.")
    result = {"status": "completed", "selection": selected, "full_validation": measured, "paired_comparison": paired,
              "development_screen": {"accuracy_path": accuracy_pass, "probability_path": probability_pass, "passed": accuracy_pass or probability_pass},
              "interpretation": "Exploratory development comparison after selection on overlapping validation data; no calibrated confidence or independent confirmation.",
              "history_unchanged": True, "calibration_test_opened": False, "product_integration": False}
    save_json(output / "results.json", result)
    point = measured["views"]["averaged"]
    lines = ["# Learning-rate sweep results", "", f"Selected {selected['candidate']} at update {selected['step']} by the fixed 100-pair monitor.", "",
             f"Full validation: {point['matches']}/311 clear matches; log-loss {point['metrics']['log_loss']['value']:.4f}; AUC {point['metrics']['auc']['value']:.4f}.",
             "", "This is development evidence after validation-based selection. Calibration and final-test data remain unopened.", "",
             "See results.json for grouped 95% intervals, paired McNemar, strength/length slices, both-order consistency, tie diagnostics, and the predefined development screen."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "execution.json" and p.name != "sweep_verification.json"}
    save_json(output / "sweep_verification.json", {"status": "passed", "artifact_sha256": artifacts, "history_after": history_after,
              "full_validation_pairs": 396, "clear_pairs": 311, "sweep_updates": 450, "probe_updates": 3, "calibration_test_opened": False})


def run(output, mode):
    plan = validate(output)
    if (output / "sweep_verification.json").exists():
        raise ValueError("This sweep is complete; preserve its sealed results.")
    lock = output / "runner.lock"
    lock.mkdir(exist_ok=False)
    try:
        launch(output, "checks", 120)
        for job in ("probe_train", "probe_resume"):
            launch(output, job, plan["budgets"]["probe_seconds_each"])
        check_probe(output)
        if mode == "check":
            return
        status(output / "execution.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
        for job, spec in plan["jobs"].items():
            markers = {p.stem.removesuffix("_worker"): read_json(p) for p in output.glob("*_worker.json")}
            used = sum(r.get("seconds", 0) for name, r in markers.items() if name.startswith("lr_"))
            candidate_used = sum(r.get("seconds", 0) for name, r in markers.items() if name.startswith(spec["candidate"] + "_"))
            limit = min(spec["wall_seconds"], plan["budgets"]["candidate_seconds"] - candidate_used, plan["budgets"]["sweep_and_full_seconds"] - used)
            if limit <= 0:
                raise TimeoutError("The preregistered sweep budget is exhausted.")
            status(output / "execution.json", {"status": "running", "active_job": job, "completed_training_seconds": used})
            launch(output, job, limit)
        select(output)
        launch(output, "full", plan["budgets"]["full_validation_seconds"])
        report(output)
        status(output / "execution.json", {"status": "completed", "results": "results.json", "report": "RESULTS.md"})
    except Exception as error:
        status(output / "execution.json", {"status": "stopped", "error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        lock.rmdir()


def recover(output, job):
    if (output / "runner.lock").exists() or job not in jobs():
        raise ValueError("Recovery requires a stopped runner and a known training segment.")
    marker = output / (job + "_worker.json")
    if read_json(marker)["status"] not in ("worker_failed", "deadline_exceeded"):
        raise ValueError("Only a recorded failed or timed-out segment can be recovered.")
    number = len(list(output.glob(job + "_attempt_*_worker.json"))) + 1
    archived = f"{job}_attempt_{number:04d}"
    if (output / job).exists():
        (output / job).rename(output / archived)
    (output / (job + ".log")).rename(output / (archived + ".log"))
    marker.rename(output / (archived + "_worker.json"))
    if (output / (job + "_failure.json")).exists():
        (output / (job + "_failure.json")).rename(output / (archived + "_failure.json"))
    print("Preserved failed attempt. An explicit run command can replay from the preceding complete checkpoint within the remaining time budget.")


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif len(sys.argv) == 4 and sys.argv[1] == "worker":
        folder, job = Path(sys.argv[2]).resolve(), sys.argv[3]
        try:
            worker(folder, job)
        except Exception as error:
            save_json(folder / (job + "_failure.json"), {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
            raise
    elif len(sys.argv) == 3 and sys.argv[1] in ("check", "run"):
        run(Path(sys.argv[2]).resolve(), sys.argv[1])
    elif len(sys.argv) == 4 and sys.argv[1] == "recover":
        folder = Path(sys.argv[2]).resolve()
        validate(folder)
        recover(folder, sys.argv[3])
    else:
        raise SystemExit("Use prepare, check RUN_FOLDER, run RUN_FOLDER, or explicit recover RUN_FOLDER JOB.")
