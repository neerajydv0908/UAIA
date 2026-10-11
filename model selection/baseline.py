"""Measure local Qwen preference decisions on validation data; never train."""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Training Data/prepared/helpsteer2_v2"
LABELS = ("A", "B", "tie_preference")
SWAP = {"A": "B", "B": "A", "tie_preference": "tie_preference"}
LIMITS = {"per_label": 20, "input_tokens": 4096, "output_tokens": 16, "wall_seconds": 900}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_sample(rows, per_label):
    counts, groups, sample = Counter(), set(), []
    ordered = sorted(rows, key=lambda row: hashlib.sha256(("baseline-v1:" + row["id"]).encode()).hexdigest())
    for row in ordered:
        label = row["target"]["preference"]
        if row["split"] != "validation" or label not in LABELS:
            raise ValueError("Only validation rows with recognized preference labels are allowed.")
        if row["group_id"] in groups or counts[label] >= per_label:
            continue
        sample.append({"id": row["id"], "group_id": row["group_id"], "target": label, "input": row["input"]})
        groups.add(row["group_id"])
        counts[label] += 1
    if any(counts[label] != per_label for label in LABELS):
        raise ValueError("Not enough distinct groups for the requested balanced sample.")
    return sample


def messages_for(row, swapped, rubric):
    source = row["input"]
    answers = source["answers"]
    payload = {"question": source["question"], "evidence": source["evidence"],
               "answers": {"A": answers["B" if swapped else "A"], "B": answers["A" if swapped else "B"]}}
    return [{"role": "system", "content": rubric}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def classify_output(raw, finish_reason, swapped):
    decision = raw.strip()
    if finish_reason != "stop":
        return "output_limit", None
    if decision not in LABELS:
        return "invalid_output", None
    return "completed", SWAP[decision] if swapped else decision


def rate_report(records):
    correct = sum(row["status"] == "completed" and row["decision"] == row["target"] for row in records)
    return {"correct": correct, "total": len(records), "agreement": correct / len(records) if records else None,
            "statuses": dict(Counter(row["status"] for row in records))}


def summarize(sample, records):
    lookup = {(row["id"], row["swapped"]): row for row in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate result records would distort the score.")
    full = []
    for row in sample:
        for swapped in (False, True):
            result = lookup.get((row["id"], swapped), {"status": "not_run", "decision": None})
            full.append({**result, "id": row["id"], "swapped": swapped, "target": row["target"]})
    original = [row for row in full if not row["swapped"]]
    reversed_rows = [row for row in full if row["swapped"]]
    valid_pairs = [(a, b) for a, b in zip(original, reversed_rows) if a["status"] == b["status"] == "completed"]
    seconds = [row["seconds"] for row in full if "seconds" in row and row["status"] in ("completed", "invalid_output", "output_limit")]
    ordered_seconds = sorted(seconds)
    rules = {}
    for name in ("always_A", "always_B", "always_tie", "longer_answer", "shorter_answer"):
        predictions = []
        for row in sample:
            a, b = len(row["input"]["answers"]["A"]), len(row["input"]["answers"]["B"])
            if name.startswith("always_"):
                decision = {"always_A": "A", "always_B": "B", "always_tie": "tie_preference"}[name]
            else:
                decision = "tie_preference" if a == b else ("A" if a > b else "B")
                if name == "shorter_answer":
                    decision = SWAP[decision]
            predictions.append({"status": "completed", "decision": decision, "target": row["target"]})
        rules[name] = rate_report(predictions)
    return {"original": rate_report(original), "swapped_mapped_back": rate_report(reversed_rows),
            "by_label_original": {label: rate_report([row for row in original if row["target"] == label]) for label in LABELS},
            "predictions_original": dict(Counter(row["decision"] or row["status"] for row in original)),
            "confusion_original": {label: dict(Counter(row["decision"] or row["status"] for row in original if row["target"] == label)) for label in LABELS},
            "swap_consistency": {"same_decision": sum(a["decision"] == b["decision"] for a, b in valid_pairs), "valid_pairs": len(valid_pairs), "planned_pairs": len(sample)},
            "both_orders_correct": sum(a["decision"] == b["decision"] == a["target"] for a, b in zip(original, reversed_rows) if a["status"] == b["status"] == "completed"),
            "latency_seconds": {"count": len(seconds), "median": statistics.median(seconds) if seconds else None,
                                "p95": ordered_seconds[(95 * len(seconds) + 99) // 100 - 1] if seconds else None},
            "simple_rules_original": rules, "unresolved": [row for row in full if row["status"] != "completed"]}


def worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    plan = json.loads((output / "plan.json").read_text())
    sample = read_rows(output / "sample.jsonl")
    choice, limits = plan["model"], plan["limits"]
    model_path = HERE / choice["local_directory"]
    receipt = json.loads((HERE / "download_manifest.json").read_text())
    if receipt["revision"] != choice["revision"]:
        raise ValueError("Model receipt and selected revision disagree.")
    for item in receipt["files"]:
        if fingerprint(model_path / item["file"]) != item["sha256"]:
            raise ValueError("Downloaded model file changed: " + item["file"])
    started = time.perf_counter()
    model, tokenizer = load(str(model_path), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    runtime = {"load_seconds": time.perf_counter() - started, "downloaded_file_hashes_verified": True}
    save_json(output / "runtime.json", runtime)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(sample):
            for swapped in ((False, True) if index % 2 == 0 else (True, False)):
                started = time.perf_counter()
                result = {"id": row["id"], "swapped": swapped, "target": row["target"], "decision": None}
                try:
                    prompt = tokenizer.apply_chat_template(messages_for(row, swapped, plan["rubric"]), tokenize=True, add_generation_prompt=True)
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
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{index + 1}/{len(sample)} swapped={swapped} status={result['status']} decision={result['decision']}", flush=True)
                mx.clear_cache()
    runtime["mlx_peak_memory_gib"] = mx.get_peak_memory() / 1024**3
    save_json(output / "runtime.json", runtime)


def main():
    rows = read_rows(DATA / "validation.jsonl")
    data_report = json.loads((DATA / "report.json").read_text())
    data_hash = fingerprint(DATA / "validation.jsonl")
    if data_hash != data_report["output_sha256"]["validation.jsonl"]:
        raise ValueError("Validation data differs from its preparation report.")
    sample = select_sample(rows, LIMITS["per_label"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = HERE / "results" / ("baseline_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    choice = json.loads((HERE / "model_choice.json").read_text())
    plan = {"kind": "balanced_validation_preference_baseline_v1", "created_utc": stamp, "model": choice,
            "limits": LIMITS, "sample_rule": "SHA256(baseline-v1: + id); first 20 per label; one row per question group; no replacement for long inputs",
            "validation_rows": len(rows), "validation_sha256": data_hash, "dataset_revision": data_report["source_revision"],
            "code_sha256": fingerprint(Path(__file__)), "environment_lock_sha256": fingerprint(HERE / "requirements.lock.txt"),
            "rubric": (HERE / "judge_instructions.txt").read_text(), "temperature": 0.0,
            "planned_decisions": 2 * len(sample), "automatic_retries": 0, "training": False, "paid_api_calls": 0,
            "calibration_or_test_opened": False, "limit_seconds_covers": "worker imports, hash verification, loading and all inference"}
    save_json(output / "plan.json", plan)
    with (output / "sample.jsonl").open("x", encoding="utf-8") as stream:
        for row in sample:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Saved plan before inference:", output, flush=True)
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(output)], timeout=LIMITS["wall_seconds"], check=False)
        state = "finished" if process.returncode == 0 else "worker_failed"
    except subprocess.TimeoutExpired:
        state = "deadline_exceeded"
    records = []
    prediction_path = output / "predictions.jsonl"
    if prediction_path.exists():
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    state = "interrupted_result_write"
                    break
    report = {"worker_state": state, "wall_seconds": time.perf_counter() - started, "metrics": summarize(sample, records),
              "interpretation": "Agreement with public human preference labels on a balanced development sample; not truth accuracy or a final test."}
    save_json(output / "summary.json", report)
    print("Saved summary:", output / "summary.json", flush=True)
    if state != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("Run baseline.py without arguments; --worker is internal.")
