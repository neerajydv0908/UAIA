"""Measure training-only lengths and prepare a bounded, answer-only LoRA trial."""
import hashlib
import json
import math
import os
import subprocess
import time
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from baseline import LABELS, SWAP, messages_for, read_rows, save_json
from baseline_guard import ROOT, SNAPSHOT, sha256, verify

DATA = ROOT / "Training Data/prepared/helpsteer2_v2"
MAX_LENGTH = 2048
TRIAL_GROUPS = 100


def training_record(row, swapped, rubric):
    if row["split"] != "train" or row["target"]["preference"] not in LABELS:
        raise ValueError("Only training rows with recognized labels are allowed.")
    target = row["target"]["preference"]
    label = SWAP[target] if swapped else target
    return {"messages": messages_for(row, swapped, rubric) + [{"role": "assistant", "content": label}]}


def tokenize_record(record, tokenizer):
    messages = record["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    if not isinstance(prompt, list) or not isinstance(tokens, list):
        raise ValueError("The tokenizer must return lists of token IDs.")
    if tokens[:len(prompt)] != prompt or len(tokens) <= len(prompt):
        raise ValueError("Training prompt is not an exact prefix of the complete conversation.")
    return tokens, len(prompt)


def length_summary(lengths):
    values = sorted(lengths)
    return {"count": len(values), "min": values[0], "max": values[-1],
            **{f"p{p}": values[math.ceil(p * len(values) / 100) - 1] for p in (50, 90, 95, 99)}}


def main():
    started = time.perf_counter()
    guard = verify()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    choice = json.loads((ROOT / SNAPSHOT / "model_choice_at_run.json").read_text())
    rubric = (ROOT / SNAPSHOT / "judge_instructions.txt").read_text()
    report = json.loads((DATA / "report.json").read_text())
    train_hash = sha256(DATA / "train.jsonl")
    if train_hash != report["output_sha256"]["train.jsonl"]:
        raise ValueError("Training data differs from the recorded release.")
    rows = read_rows(DATA / "train.jsonl")
    baseline_groups = {row["group_id"] for row in read_rows(ROOT / SNAPSHOT / "sample.jsonl")}
    if any(row["group_id"] in baseline_groups for row in rows):
        raise ValueError("A training group overlaps the frozen baseline.")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model selection" / choice["local_directory"], local_files_only=True, trust_remote_code=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "model selection/results" / ("training_readiness_" + stamp)
    output.mkdir(exist_ok=False)
    lengths, eligible, probes, longest, shortest = [], [], {}, 0, MAX_LENGTH + 1
    with (output / "lengths.jsonl").open("x") as stream:
        for row in rows:
            records = [training_record(row, swapped, rubric) for swapped in (False, True)]
            encoded = [tokenize_record(record, tokenizer) for record in records]
            sizes = [len(tokens) for tokens, offset in encoded]
            fits = max(sizes) <= MAX_LENGTH
            entry = {"id": row["id"], "group_id": row["group_id"], "label": row["target"]["preference"], "lengths": sizes, "eligible": fits}
            stream.write(json.dumps(entry) + "\n")
            lengths.extend(sizes)
            if fits:
                eligible.append((row, records))
                if min(sizes) < shortest:
                    shortest = min(sizes)
                    probes["shortest"] = records[sizes.index(shortest)]
                if max(sizes) > longest:
                    longest = max(sizes)
                    probes["longest"] = records[sizes.index(longest)]
    eligible.sort(key=lambda item: hashlib.sha256(("training-trial-v1:" + item[0]["id"]).encode()).hexdigest())
    selected, groups = [], set()
    for row, records in eligible:
        if row["group_id"] not in groups and len(groups) < TRIAL_GROUPS:
            selected.append((row, records))
            groups.add(row["group_id"])
    if len(groups) != TRIAL_GROUPS:
        raise ValueError("Too few eligible training groups for this trial.")
    for name, collection in (("train_eligible.jsonl", eligible), ("trial_train.jsonl", selected)):
        with (output / name).open("x") as stream:
            for row, records in collection:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    save_json(output / "trial_index.json", [{"id": row["id"], "group_id": row["group_id"], "label": row["target"]["preference"]} for row, records in selected])
    save_json(output / "probe_records.json", probes)
    config = {"model": choice["local_directory"], "model_revision": choice["revision"], "batch_size": 1,
              "max_seq_length": MAX_LENGTH, "lora_layers": 4, "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
              "mask_prompt": True, "grad_checkpoint": True, "optimizer": "Adam", "learning_rate": 0.00001,
              "seed": 20261006, "probe_wall_seconds": 600, "mlx_memory_limit_gib": 18, "probe_optimizer_updates": 0,
              "future_trial_iterations": 20, "future_trial_wall_seconds": 900, "future_trial_status": "proposed_not_run"}
    save_json(output / "training_config.json", config)
    profile = {"status": "prepared", "wall_seconds": time.perf_counter() - started, "baseline_guard": guard, "train_rows": len(rows), "train_sha256": train_hash,
               "lengths_both_orders_with_target": length_summary(lengths), "max_seq_length": MAX_LENGTH,
               "eligible_pairs": len(eligible), "excluded_pairs": len(rows) - len(eligible), "truncated_examples": 0,
               "eligible_labels": dict(Counter(row["target"]["preference"] for row, records in eligible)),
               "trial_groups": len(groups), "trial_presentations": 2 * len(selected),
               "trial_original_labels": dict(Counter(row["target"]["preference"] for row, records in selected)),
               "calibration_or_test_opened": False, "optimizer_updates": 0, "code_sha256": sha256(Path(__file__)),
               "files_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    save_json(output / "profile.json", profile)
    print(json.dumps(profile, indent=2))
    print("Prepared:", output, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        main()
    elif not sys.argv[1:]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"], check=True, timeout=600)
    else:
        raise SystemExit("Run prepare_training.py without arguments; --worker is internal.")
