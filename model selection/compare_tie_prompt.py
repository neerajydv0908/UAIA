"""Compare one tie-instruction change using fixed original and adapter weights."""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from baseline import classify_output, messages_for, read_rows, save_json, summarize
from baseline_guard import ROOT, SNAPSHOT, sha256, verify
from expand_trial import paired_changes
from train_trial import adapter_digest, collect_predictions

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "results/error_review_20261006T221309230086Z"
TRAINED = HERE / "results/expanded_trial_20261006T132929116227Z"
FIRST = HERE / "results/training_trial_20261006T104025828291Z"
MODELS = ("baseline", "updates200")


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_receipt(folder, name):
    receipt = json.loads((folder / name).read_text())
    if receipt["status"] != "passed":
        raise ValueError("Earlier verification did not pass.")
    for path, expected in receipt["artifact_sha256"].items():
        if sha256(folder / path) != expected:
            raise ValueError("Earlier artifact changed: " + path)
    return {"files_checked": len(receipt["artifact_sha256"]), "receipt_sha256": sha256(folder / name)}


def history_guard():
    return {"baseline": verify(), "first_trial": verify_receipt(FIRST, "trial_verification.json"),
            "expanded_trial": verify_receipt(TRAINED, "expanded_verification.json"),
            "error_review": verify_receipt(REVIEW, "review_verification.json")}


def validate_proposal(proposal, original):
    old = "Use tie_preference when their overall answer quality is equal."
    new = "Use tie_preference when their overall answer quality is about the same."
    if proposal["change_only"] != {"old_sentence": old, "new_sentence": new} or original.count(old) != 1:
        raise ValueError("Require the previously proposed single sentence change.")
    if proposal["candidate_rubric"] != original.replace(old, new):
        raise ValueError("The candidate changed more than the selected tie sentence.")
    if text_hash(original) != proposal["original_rubric_sha256"] or text_hash(proposal["candidate_rubric"]) != proposal["candidate_rubric_sha256"]:
        raise ValueError("Prompt fingerprints differ from the proposal.")
    expected = {"planned_new_decisions": 240, "question_groups": 60, "answer_orders_per_group": 2,
                "per_model_worker_wall_seconds": 900, "max_input_tokens": 4096, "max_output_tokens": 16,
                "temperature": 0, "automatic_retries": 0, "new_training_updates": 0, "paid_api_calls": 0,
                "calibration_or_test_opened": False}
    if any(proposal[key] != value for key, value in expected.items()):
        raise ValueError("The previously chosen experiment limits changed.")


def checked_summary(sample, records):
    labels = {row["id"]: row["target"] for row in sample}
    for row in records:
        if row["id"] not in labels or type(row["swapped"]) is not bool or row["target"] != labels[row["id"]]:
            raise ValueError("Unexpected prediction identity, orientation, or answer key.")
    return summarize(sample, records)


def acceptance_screen(metrics, records, rules):
    after = metrics["updates200"]
    correct = lambda swapped, tie: sum(row["status"] == "completed" and row["decision"] == row["target"]
                 for row in records["updates200"] if row["swapped"] == swapped and (row["target"] == "tie_preference") == tie)
    actual = {"original_tie_matches_min": correct(False, True), "swapped_tie_matches_min": correct(True, True),
              "original_clear_matches_min": correct(False, False), "swapped_clear_matches_min": correct(True, False),
              "order_consistency_min": after["swap_consistency"]["same_decision"], "both_orders_match_min": after["both_orders_correct"],
              "original_total_matches_min": after["original"]["correct"], "swapped_total_matches_min": after["swapped_mapped_back"]["correct"]}
    checks = {key: {"actual": actual[key], "minimum": minimum, "passed": actual[key] >= minimum}
              for key, minimum in rules["on_200_update_adapter"].items()}
    valid = all(len(records[name]) == 120 and not metrics[name]["unresolved"] for name in MODELS)
    return {"all_240_planned_decisions_valid": valid, "thresholds": checks,
            "passed": valid and all(check["passed"] for check in checks.values())}


def worker(output, name):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    plan = json.loads((output / "plan.json").read_text())
    if history_guard() != plan["history_before"]:
        raise ValueError("An earlier experiment or receipt changed after planning.")
    for filename, expected in plan["code_sha256"].items():
        if sha256(HERE / filename) != expected:
            raise ValueError("Experiment source changed after planning.")
    proposal = plan["proposal"]
    validate_proposal(proposal, plan["original_rubric"])
    if sha256(ROOT / SNAPSHOT / "sample.jsonl") != proposal["validation_sample_sha256"]:
        raise ValueError("The fixed sample changed.")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.utils import load_adapters
    model, tokenizer = load(str(HERE / plan["model"]["local_directory"]), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    adapter_before = None
    if name == "updates200":
        load_adapters(model, str(TRAINED / "adapter"))
        adapter_before = adapter_digest([(key, value) for key, value in tree_flatten(model.parameters()) if key.endswith((".lora_a", ".lora_b"))])
        if adapter_before != json.loads((TRAINED / "training_result.json").read_text())["adapter_digest_after"]:
            raise ValueError("Loaded adapter numbers differ from the measured checkpoint.")
    model.freeze()
    model.eval()
    sample = read_rows(ROOT / SNAPSHOT / "sample.jsonl")
    with (output / name / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(sample):
            for swapped in ((False, True) if index % 2 == 0 else (True, False)):
                started = time.perf_counter()
                result = {"id": row["id"], "swapped": swapped, "target": row["target"], "decision": None}
                try:
                    messages = messages_for(row, swapped, proposal["candidate_rubric"])
                    result["messages_sha256"] = text_hash(json.dumps(messages, ensure_ascii=False))
                    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    result["input_tokens"] = len(prompt)
                    if len(prompt) > proposal["max_input_tokens"]:
                        result["status"] = "input_too_long"
                    else:
                        raw, finish, output_tokens = "", None, 0
                        for piece in stream_generate(model, tokenizer, prompt=prompt, max_tokens=proposal["max_output_tokens"], sampler=make_sampler(temp=0.0)):
                            raw += piece.text
                            finish, output_tokens = piece.finish_reason, piece.generation_tokens
                        result.update(raw_output=raw, finish_reason=finish, output_tokens=output_tokens)
                        result["status"], result["decision"] = classify_output(raw, finish, swapped)
                except Exception as error:
                    result.update(status="error", error_type=type(error).__name__)
                result.update(seconds=time.perf_counter() - started, mlx_peak_memory_gib=mx.get_peak_memory() / 1024**3)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{name}: question {index + 1}/60, swapped={swapped}, {result['status']}", flush=True)
                mx.clear_cache()
    adapter_after = adapter_before
    if name == "updates200":
        adapter_after = adapter_digest([(key, value) for key, value in tree_flatten(model.parameters()) if key.endswith((".lora_a", ".lora_b"))])
    if adapter_before != adapter_after:
        raise ValueError("Adapter numbers changed during inference.")
    save_json(output / name / "runtime.json", {"mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3,
              "adapter_digest_before": adapter_before, "adapter_digest_after": adapter_after,
              "model_frozen": True, "training_updates": 0, "weight_files_written": False})


def supervise(output, name, seconds):
    result = {"status": "running", "wall_limit_seconds": seconds}
    save_json(output / name / "worker.json", result)
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(output), name], timeout=seconds, check=False)
        result["status"] = "completed" if process.returncode == 0 else "worker_failed"
    except subprocess.TimeoutExpired:
        result["status"] = "deadline_exceeded"
    result["wall_seconds"] = time.perf_counter() - started
    save_json(output / name / "worker.json", result)
    return result


def main():
    guards = history_guard()
    proposal = json.loads((REVIEW / "next_experiment_plan.json").read_text())
    baseline_plan = json.loads((ROOT / SNAPSHOT / "plan.json").read_text())
    validate_proposal(proposal, baseline_plan["rubric"])
    if sha256(HERE / "requirements.lock.txt") != baseline_plan["environment_lock_sha256"]:
        raise ValueError("Pinned runtime requirements changed.")
    output = HERE / "results" / ("tie_prompt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    for name in MODELS:
        (output / name).mkdir()
    plan = {"kind": "fixed_weights_tie_prompt_comparison", "proposal": proposal, "history_before": guards,
            "model": baseline_plan["model"], "original_rubric": baseline_plan["rubric"],
            "proposal_sha256": sha256(REVIEW / "next_experiment_plan.json"),
            "code_sha256": {name: sha256(HERE / name) for name in ("compare_tie_prompt.py", "baseline.py", "baseline_guard.py", "expand_trial.py", "train_trial.py")}}
    save_json(output / "plan.json", plan)
    print("Prompt comparison output:", output, flush=True)
    workers = {name: supervise(output, name, proposal["per_model_worker_wall_seconds"]) for name in MODELS}
    sample = read_rows(ROOT / SNAPSHOT / "sample.jsonl")
    records, damaged, metrics, comparisons = {}, {}, {}, {}
    for name in MODELS:
        records[name], damaged[name] = collect_predictions(output / name / "predictions.jsonl")
        metrics[name] = checked_summary(sample, records[name])
        before = read_rows(ROOT / proposal["existing_controls"][name])
        comparisons[name] = {"old_prompt": checked_summary(sample, before), "candidate_prompt": metrics[name],
                             "paired_changes": {str(swapped): paired_changes(before, records[name], swapped) for swapped in (False, True)}}
    screen = acceptance_screen(metrics, records, proposal["development_acceptance_screen"])
    after = history_guard()
    complete = all(row["status"] == "completed" for row in workers.values()) and not any(damaged.values()) and after == guards
    save_json(output / "comparison.json", {"workers": workers, "damaged_prediction_tails": damaged,
              "comparisons": comparisons, "development_screen": screen, "execution_complete": complete,
              "history_after": after, "candidate_passes_run_and_screen": complete and screen["passed"],
              "interpretation": "Repeated development preference comparison with fixed weights; no training, final test, or factual-accuracy claim."})
    print("Comparison saved:", output / "comparison.json", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 4 and sys.argv[1] == "--worker" and sys.argv[3] in MODELS:
        worker(Path(sys.argv[2]).resolve(), sys.argv[3])
    else:
        raise SystemExit("Run compare_tie_prompt.py without arguments.")
