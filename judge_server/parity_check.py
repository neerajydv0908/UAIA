"""Prove the served judge reads prompts exactly as in training, then (optionally) spot-check scores.

Token check needs only the tokenizer. Uses validation rows only; never calibration or test.

  "model selection/.colab-venv/bin/python" -m judge_server.parity_check            # tokens, 20 rows
  "model selection/.colab-venv/bin/python" -m judge_server.parity_check --score 20 # + model accuracy
"""

import argparse
import json
import sys
from pathlib import Path

from .server import ADAPTER_DIR, BASE_DIR, RUBRIC, ROOT, Judge, chat_ids, load_tokenizer, messages

BUNDLE = ROOT / "model selection/results/colab_preparation_20261007/bundle"
RAW = ROOT / "Training Data/prepared/helpsteer2_v2/validation.jsonl"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--score", type=int, default=0, help="Also run the model on this many clear pairs")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    args = parser.parse_args()

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest["rubric"] != RUBRIC:
        sys.exit("FAIL: served rubric differs from the bundle rubric")
    expected = {row["id"]: row for row in read_rows(args.bundle / "validation.jsonl")}
    every = [row for row in read_rows(args.raw) if row["id"] in expected]
    raw = every[: args.rows]
    tokenizer = load_tokenizer(BASE_DIR)
    mismatches = 0
    for row in raw:
        source = row["input"]
        for swapped in (False, True):
            a, b = (source["answers"]["B"], source["answers"]["A"]) if swapped else (source["answers"]["A"], source["answers"]["B"])
            ids = chat_ids(tokenizer, messages(source["question"], a, b, source["evidence"]))
            if ids != expected[row["id"]]["tokens"]["json"][int(swapped)]:
                mismatches += 1
    print(f"Token parity: {2 * len(raw) - mismatches}/{2 * len(raw)} prompts identical to the training bundle")
    if mismatches:
        sys.exit("FAIL: served prompts differ from training prompts")

    if args.score:
        judge = Judge(BASE_DIR, args.adapter)
        clear = [row for row in every if expected[row["id"]]["target"] in ("A", "B")][: args.score]
        correct = 0
        for row in clear:
            source, label = row["input"], expected[row["id"]]["target"]
            result = judge.compare(source["question"], source["answers"]["A"], source["answers"]["B"])
            correct += (result["p_a"] > .5) == (label == "A")
            print(f"{row['id']}: p_a={result['p_a']:.3f} label={label} ({result['seconds']}s)")
        print(f"Spot check: {correct}/{len(clear)} agree with human labels "
              f"(validation accuracy for this adapter was measured on all 311 clear pairs)")


if __name__ == "__main__":
    main()
