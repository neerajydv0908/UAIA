"""Bounded, inference-only baseline v2; historical runners stay unchanged."""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

from baseline import messages_for, read_rows, save_json
from baseline_guard import ROOT, SNAPSHOT, sha256
from compare_tie_prompt import FIRST, TRAINED, history_guard, verify_receipt
from train_trial import adapter_digest
from judge_head import project_hidden

HERE = Path(__file__).resolve().parent
DATA = ROOT / "Training Data/prepared/helpsteer2_v2"
JOBS = {"baseline_v2_json": (None, "json"), "adapter20_json": (FIRST, "json"),
        "adapter200_json": (TRAINED, "json"), "baseline_v2_plain": (None, "plain")}
TOKENS = {"A": 32, "B": 33, "tie_prefix": 48731}
WALL_SECONDS = 2400
MAX_INPUT = 4096


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_rubric(original):
    return original.replace("A, B, or tie_preference", "A or B").replace(
        " Use tie_preference when their overall answer quality is equal.", "").replace(
        "A preference tie alone", "A preference alone")


def readable_turns(text):
    return re.sub(r"<extra_id_1>(User|Assistant)\n?", lambda m: "\n" + m[1] + ":\n", text).strip()


def messages(row, swapped, style, rubric):
    if style == "json":
        return messages_for(row, swapped, rubric)
    if style != "plain" or row["input"]["evidence"]:
        raise ValueError("Plain preference format requires empty evidence and a known style.")
    source = row["input"]
    a, b = (source["answers"][key] for key in (("B", "A") if swapped else ("A", "B")))
    content = "### Question\n" + readable_turns(source["question"])
    content += "\n\n### Answer A\n" + a + "\n\n### Answer B\n" + b
    return [{"role": "system", "content": rubric}, {"role": "user", "content": content}]


def probabilities(logits):
    if len(logits) != 3 or not all(math.isfinite(x) for x in logits):
        raise ValueError("Require finite A, B, and tie-prefix logits.")
    weights = [math.exp(value - max(logits[:2])) for value in logits[:2]]
    binary = weights[0] / sum(weights)
    weights3 = [math.exp(value - max(logits)) for value in logits]
    return binary, [value / sum(weights3) for value in weights3]


def prepare():
    history = history_guard()
    history["prompt_trial"] = verify_receipt(HERE / "results/tie_prompt_20261006T230223795161Z", "prompt_verification.json")
    environment = {}
    for line in (HERE / "requirements.lock.txt").read_text().splitlines():
        name, expected = line.split("==")
        environment[name] = version(name)
        if environment[name] != expected:
            raise ValueError("Installed dependency differs from the pinned environment: " + name)
    release = json.loads((DATA / "report.json").read_text())
    for split in ("train", "validation"):
        if sha256(DATA / (split + ".jsonl")) != release["output_sha256"][split + ".jsonl"]:
            raise ValueError("The prepared data changed.")
    train, validation = (read_rows(DATA / (split + ".jsonl")) for split in ("train", "validation"))
    if {r["group_id"] for r in train} & {r["group_id"] for r in validation}:
        raise ValueError("Training and validation question groups overlap.")
    sample = [{"id": r["id"], "group_id": r["group_id"], "target": r["target"]["preference"],
               "strength": abs(r["target"]["raw_preference_strength"]), "input": r["input"]} for r in validation]
    if len(sample) != 396 or Counter(r["target"] for r in sample) != {"A": 140, "B": 171, "tie_preference": 85}:
        raise ValueError("Unexpected validation release counts.")
    original = (ROOT / SNAPSHOT / "judge_instructions.txt").read_text()
    rubric = binary_rubric(original)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HERE / "models/Qwen3-4B-Instruct-2507", local_files_only=True, trust_remote_code=False)
    encoding = {text: tokenizer.encode(text, add_special_tokens=False) for text in ("A", "B", "tie", "tie_preference")}
    if encoding != {"A": [32], "B": [33], "tie": [48731], "tie_preference": [48731, 92718]}:
        raise ValueError("Decision-token identities changed.")
    lengths = []
    for row in sample:
        for style in ("json", "plain"):
            for swapped in (False, True):
                msg = messages(row, swapped, style, rubric)
                ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=False)
                for label in ("A", "B"):
                    full = tokenizer.apply_chat_template(msg + [{"role": "assistant", "content": label}], tokenize=True, return_dict=False)
                    if full[:len(ids)] != ids or full[len(ids)] != TOKENS[label]:
                        raise ValueError("Decision position is not the expected token.")
                lengths.append({"id": row["id"], "style": style, "swapped": swapped, "tokens": len(ids),
                                "messages_sha256": digest_text(json.dumps(msg, ensure_ascii=False))})
    if max(row["tokens"] for row in lengths) > MAX_INPUT:
        raise ValueError("Full validation does not fit the planned input limit; do not truncate.")
    output = HERE / "results" / ("baseline_v2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    with (output / "sample.jsonl").open("x") as stream:
        for row in sample:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(output / "input_checks.json", {"token_ids": encoding, "lengths": lengths})
    save_json(output / "environment.json", environment)
    tie_votes = {}
    for split, rows in (("train", train), ("validation", validation)):
        votes = [{v["strength"] for v in r["audit_only"]["all_preferences_unprocessed"]} for r in rows if r["target"]["preference"] == "tie_preference"]
        tie_votes[split] = {"tie_rows": len(votes), "opposing_votes": sum(min(v) < 0 < max(v) for v in votes), "with_zero_vote": sum(0 in v for v in votes)}
    settings = {"bootstrap_replicates": 10000, "bootstrap_seed": 20261007, "interval": "95% percentile, resample question groups",
                "minimum_accuracy_gain": 0.03, "minimum_logloss_reduction": 0.02, "familywise_alpha": 0.05,
                "format_rule": "Adopt plain if accuracy gain >= .03, paired accuracy CI lower > 0, Holm McNemar p < .05, and log-loss does not rise; OR log-loss reduction >= .02, paired log-loss CI upper < 0, and accuracy does not fall. Otherwise retain JSON.",
                "tie_bands": [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2], "minimum_clear_coverage_for_tie_band": 0.95,
                "tie_band_rule": "Maximize (tie recall + clear coverage)/2 within the coverage constraint; break equal scores toward the smaller band. Validation-fitted diagnostic only.",
                "primary_comparisons": [["baseline_v2_json", "adapter20_json"], ["baseline_v2_json", "adapter200_json"], ["baseline_v2_json", "baseline_v2_plain"]]}
    plan = {"kind": "baseline_v2_full_validation_binary_probabilities", "status": "planned_before_inference",
            "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554", "dataset_revision": release["source_revision"],
            "jobs": {key: {"adapter": str(folder.relative_to(ROOT)) if folder else None, "format": style} for key, (folder, style) in JOBS.items()},
            "rubric": rubric, "rubric_sha256": digest_text(rubric), "historical_rubric_sha256": digest_text(original),
            "sample_sha256": sha256(output / "sample.jsonl"), "input_checks_sha256": sha256(output / "input_checks.json"),
            "validation_sha256": release["output_sha256"]["validation.jsonl"], "train_sha256": release["output_sha256"]["train.jsonl"],
            "validation_pairs": 396, "validation_groups": len({r["group_id"] for r in sample}), "clear_pairs": 311, "tie_pairs": 85,
            "planned_score_reads": 3168, "score_reads_per_job": 792, "probe_score_evaluations": 2, "worker_wall_seconds": WALL_SECONDS,
            "max_input_tokens": MAX_INPUT, "prefill_chunk_tokens": 512, "automatic_retries": 0, "training_updates": 0,
            "paid_api_calls": 0, "calibration_or_test_opened": False, "token_ids": TOKENS, "statistics": settings,
            "history_before": history, "tie_vote_audit": tie_votes, "split_counts_from_existing_report": release["split_counts"],
            "legacy_eligible_ab_pairs": 5037, "length_definition": "Unicode character count; equal lengths excluded from shorter-winner slice",
            "code_sha256": {name: sha256(HERE / name) for name in ("judge_v2.py", "judge_v2_metrics.py", "test_judge_v2.py")},
            "reused_code_sha256": {name: sha256(HERE / name) for name in ("baseline.py", "baseline_guard.py", "train_trial.py", "compare_tie_prompt.py")}}
    save_json(output / "plan.json", plan)
    print("Prepared v2 milestone:", output, flush=True)


def first_logits(model, ids, chunk_size, token_ids=None):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    if not ids or chunk_size < 1:
        raise ValueError("Require a nonempty prompt and a positive prefill chunk size.")
    cache = make_prompt_cache(model)
    for offset in range(0, len(ids) - 1, chunk_size):
        model.model(mx.array([ids[offset:min(offset + chunk_size, len(ids) - 1)]]), cache=cache)
        mx.eval([part.state for part in cache])
        mx.clear_cache()
    hidden = model.model(mx.array([ids[-1:]]), cache=cache)[0, -1]
    logits = project_hidden(model, hidden, token_ids=token_ids)
    mx.eval(logits)
    return logits


def worker(output, job):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    plan = json.loads((output / "plan.json").read_text())
    if sha256(output / "sample.jsonl") != plan["sample_sha256"] or sha256(output / "input_checks.json") != plan["input_checks_sha256"]:
        raise ValueError("Prepared inference inputs changed.")
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters
    model, tokenizer = load(str(HERE / "models/Qwen3-4B-Instruct-2507"), tokenizer_config={"trust_remote_code": False, "local_files_only": True})
    folder, style = JOBS[job]
    before = None
    if folder:
        load_adapters(model, str(folder / "adapter"))
        before = adapter_digest([(k, v) for k, v in tree_flatten(model.parameters()) if k.endswith((".lora_a", ".lora_b"))])
        if before != json.loads((folder / "training_result.json").read_text())["adapter_digest_after"]:
            raise ValueError("Loaded adapter differs from the measured checkpoint.")
    model.freeze()
    model.eval()
    if job == "baseline_v2_json":
        ids = tokenizer.encode("Question: which number is larger, 2 or 3? Answer:")
        direct = project_hidden(model, model.model(mx.array([ids]))[0, -1], token_ids=None)
        mx.eval(direct)
        cached = first_logits(model, ids, 7)
        error = mx.max(mx.abs(direct - cached)).item()
        direct_p = probabilities(direct[mx.array(list(TOKENS.values()))].tolist())[0]
        cached_p = probabilities(cached[mx.array(list(TOKENS.values()))].tolist())[0]
        probability_error = abs(direct_p - cached_p)
        passed = error <= 0.25 and probability_error <= 0.01
        save_json(output / "cache_probe.json", {"max_logit_difference": error, "logit_tolerance": 0.25,
                  "binary_probability_difference": probability_error, "probability_tolerance": 0.01,
                  "status": "passed" if passed else "failed", "score_evaluations": 2})
        if not passed:
            raise ValueError("Chunked inference disagrees with direct inference beyond the BF16 probe tolerance.")
    expected = {(r["id"], r["style"], r["swapped"]): r for r in json.loads((output / "input_checks.json").read_text())["lengths"]}
    failures = 0
    with (output / job / "scores.jsonl").open("x") as stream:
        for index, row in enumerate(read_rows(output / "sample.jsonl")):
            for swapped in ((False, True) if index % 2 == 0 else (True, False)):
                started = time.perf_counter()
                result = {"id": row["id"], "swapped": swapped, "status": "running"}
                try:
                    msg = messages(row, swapped, style, plan["rubric"])
                    fingerprint = digest_text(json.dumps(msg, ensure_ascii=False))
                    ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
                    entry = expected[row["id"], style, swapped]
                    if len(ids) != entry["tokens"] or fingerprint != entry["messages_sha256"] or len(ids) > MAX_INPUT:
                        raise ValueError("Input differs from preflight.")
                    logits = first_logits(model, ids, plan["prefill_chunk_tokens"])
                    selected = logits[mx.array(list(TOKENS.values()))].tolist()
                    binary, three_way = probabilities(selected)
                    full_logp = (logits[mx.array(list(TOKENS.values()))] - mx.logsumexp(logits)).tolist()
                    top = mx.argmax(logits).item()
                    result.update(status="completed", input_tokens=len(ids), messages_sha256=fingerprint,
                                  logits=selected, full_vocab_logprobs=full_logp, three_way_prefix_probs=three_way,
                                  p_display_a=binary, top_token_id=top, top_token_text=tokenizer.decode([top]))
                except Exception as error:
                    result.update(status="error", error_type=type(error).__name__)
                    failures += 1
                result.update(seconds=time.perf_counter() - started, mlx_peak_memory_gib=mx.get_peak_memory() / 1024**3)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                mx.clear_cache()
                if failures >= 3:
                    raise RuntimeError("Three failed score reads; stop this worker without retries.")
            if (index + 1) % 10 == 0 or index == 395:
                print(f"{job}: {index + 1}/396 pairs, failures={failures}", flush=True)
    after = adapter_digest([(k, v) for k, v in tree_flatten(model.parameters()) if k.endswith((".lora_a", ".lora_b"))]) if folder else None
    if before != after:
        raise ValueError("Adapter numbers changed during inference.")
    save_json(output / job / "runtime.json", {"adapter_digest_before": before, "adapter_digest_after": after,
              "training_updates": 0, "mlx_peak_memory_gib": mx.get_peak_memory() / 1024**3})


def run(output):
    plan = json.loads((output / "plan.json").read_text())
    for name, expected in plan["code_sha256"].items():
        if sha256(HERE / name) != expected:
            raise ValueError("Milestone source changed after preregistration.")
    workers = {}
    for job in JOBS:
        (output / job).mkdir(exist_ok=False)
        started = time.perf_counter()
        save_json(output / job / "worker.json", {"status": "running", "wall_limit_seconds": WALL_SECONDS})
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", str(output), job], timeout=WALL_SECONDS, check=False)
            status = "completed" if process.returncode == 0 else "worker_failed"
        except subprocess.TimeoutExpired:
            status = "deadline_exceeded"
        workers[job] = {"status": status, "wall_seconds": time.perf_counter() - started, "wall_limit_seconds": WALL_SECONDS}
        save_json(output / job / "worker.json", workers[job])
    save_json(output / "execution.json", {"workers": workers, "status": "completed" if all(r["status"] == "completed" for r in workers.values()) else "incomplete"})
    print("All planned workers ended:", output, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 4 and sys.argv[1] == "worker" and sys.argv[3] in JOBS:
        worker(Path(sys.argv[2]).resolve(), sys.argv[3])
    else:
        raise SystemExit("Use judge_v2.py prepare, then judge_v2.py run RESULT_FOLDER.")
