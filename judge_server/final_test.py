"""One-time final test of the locked judge on the reserved test split.

The test split was never opened during development. This script reads only
`Training Data/prepared/helpsteer2_v2/test.jsonl` (never the calibration split), scores every
pair in both answer orders with the same prompt, tokens and FP32 A/B head that the validation
reports used, and reports accuracy, log-loss and AUC on the clear pairs with 95% question-group
bootstrap intervals (10,000 replicates, seed 20261007) computed by the validation reports' own
metric code. It prints and saves aggregate numbers only, never question or answer text.

Run it once, after every development choice is final:

  "model selection/.colab-venv/bin/python" -m judge_server.final_test --smoke 3   # 3 validation pairs; never reads test
  "model selection/.colab-venv/bin/python" -m judge_server.final_test             # the real run, about 1 hour on an M1 Pro
  "model selection/.colab-venv/bin/python" -m judge_server.final_test --baseline  # also the unchanged Qwen3-4B, about 2 hours

If it stops part-way, run the same command again: it resumes from the saved scores and reports
only when every pair is scored. A finished run is never repeated.
"""

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .server import ADAPTER_DIR, BASE_DIR, MAX_TOKENS, RUBRIC, RUBRIC_SHA256, ROOT, chat_ids, file_sha256, messages

DATA = ROOT / "Training Data/prepared/helpsteer2_v2"
TEST = DATA / "test.jsonl"
VALIDATION = DATA / "validation.jsonl"
OUT = ROOT / "model selection/results/final_test"
SUMMARY = ROOT / "model selection/FINAL_TEST_RESULTS.md"
LOCKED_ADAPTER_SHA256 = "7dfd28fd29924d3655dbeb730452541723f132b6daa502dc4a745e2f7b625f5e"   # checkpoint_0150
SETTINGS = {"bootstrap_replicates": 10000, "bootstrap_seed": 20261007}   # as in model selection/colab/report.py
VALIDATION_RESULT = "252/311 clear pairs (81.0%, 95% interval 76.6–85.2%), log-loss 0.450, AUC 0.871"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pair_record(row, tokenize, score, baseline_score=None, reference=None, max_tokens=MAX_TOKENS) -> dict:
    """Score one pair in both answer orders. The record holds labels and probabilities, no text."""
    source = row["input"]
    label = row["target"]["preference"]
    a_len, b_len = (len(source["answers"][key]) for key in ("A", "B"))
    record = {"id": row["id"], "group_id": row["group_id"], "target": label,
              "strength": abs(row["target"]["raw_preference_strength"]),
              "shorter_won": ((a_len < b_len) == (label == "A")) if label in ("A", "B") and a_len != b_len else None,
              "multi_turn": "<extra_id_1>" in source["question"]}
    prompts = []
    for swapped in (False, True):
        first, second = (source["answers"]["B"], source["answers"]["A"]) if swapped else (
            source["answers"]["A"], source["answers"]["B"])
        chat = messages(source["question"], first, second, source["evidence"])
        if reference is not None and chat != reference(row, swapped):
            raise ValueError("Prompt differs from the training format; stopping before any score is recorded")
        prompts.append(tokenize(chat))
    record["tokens"] = [len(ids) for ids in prompts]
    if not all(0 < n <= max_tokens for n in record["tokens"]):
        return record | {"status": "too_long"}      # excluded and counted; never truncated

    def both_orders(fn):
        shown_a_first, shown_b_first = (float(fn(ids)) for ids in prompts)
        if not all(0 <= p <= 1 for p in (shown_a_first, shown_b_first)):
            raise RuntimeError("Judge produced a probability outside [0, 1]")
        return shown_a_first, 1 - shown_b_first     # both as P(original answer A is better)

    original, swapped = both_orders(score)
    record |= {"status": "completed", "original": original, "swapped": swapped, "p_a": (original + swapped) / 2}
    if baseline_score is not None:
        b_original, b_swapped = both_orders(baseline_score)
        record |= {"baseline_original": b_original, "baseline_swapped": b_swapped,
                   "baseline_p_a": (b_original + b_swapped) / 2}
    return record


def summarize(records: list[dict], with_baseline: bool) -> dict:
    """Aggregate metrics with the validation reports' own code (model selection/colab/metrics.py)."""
    sys.path.insert(0, str(ROOT / "model selection/colab"))
    from metrics import choice, compare, measure

    done = [r | {"averaged": r["p_a"]} for r in records if r["status"] == "completed"]
    clear = [r for r in done if r["target"] in ("A", "B")]
    ties = [r for r in done if r["target"] == "tie_preference"]

    def views(prefix=""):
        rows = [r | {"original": r[prefix + "original"], "swapped": r[prefix + "swapped"],
                     "averaged": r[prefix + "p_a"]} for r in clear]
        return {field: measure(rows, field, SETTINGS) for field in ("original", "swapped", "averaged")}

    def agreement(prefix=""):
        return sum(choice(r[prefix + "original"]) == choice(r[prefix + "swapped"]) for r in done) / len(done) if done else None

    report = {
        "pairs": len(records), "completed": len(done), "clear_pairs": len(clear), "tie_pairs": len(ties),
        "excluded_too_long": sum(r["status"] == "too_long" for r in records),
        "groups": len({r["group_id"] for r in done}),
        "judge": views(),
        "strength": {str(k): measure([r for r in clear if r["strength"] == k], "averaged", SETTINGS) for k in (1, 2, 3)},
        "shorter_winner": measure([r for r in clear if r["shorter_won"] is True], "averaged", SETTINGS),
        "turns": {str(k): measure([r for r in clear if r["multi_turn"] is k], "averaged", SETTINGS) for k in (False, True)},
        "order_agreement": agreement(),
        "ties": {"pairs": len(ties), "mean_p_a": sum(r["p_a"] for r in ties) / len(ties) if ties else None,
                 "mean_distance_from_half": sum(abs(r["p_a"] - .5) for r in ties) / len(ties) if ties else None},
        "settings": SETTINGS,
        "interpretation": "Single run on the reserved test split; 95% question-group bootstrap intervals.",
    }
    if with_baseline:
        report["baseline"] = views("baseline_")
        report["baseline_order_agreement"] = agreement("baseline_")
        before = [r | {"averaged": r["baseline_p_a"]} for r in done]
        report["paired_comparison"] = compare(before, done, SETTINGS)
    return report


def _fmt(metric, percent=False):
    if not metric or metric.get("value") is None:
        return "n/a"
    value, ci = metric["value"], metric.get("ci95")
    if percent:
        return f"{value:.1%}" + (f" (95% interval {ci[0]:.1%}–{ci[1]:.1%})" if ci else "")
    return f"{value:.3f}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "")


def summary_markdown(report: dict, run: dict) -> str:
    judge = report["judge"]["averaged"]
    m = judge["metrics"]
    rows = [f"| Fine-tuned judge (`checkpoint_0150`) | {judge['matches']}/{report['clear_pairs']}, "
            f"{_fmt(m['accuracy'], True)} | {_fmt(m['log_loss'])} | {_fmt(m['auc'])} |"]
    if "baseline" in report:
        base = report["baseline"]["averaged"]
        bm = base["metrics"]
        rows.insert(0, f"| Unchanged Qwen3-4B | {base['matches']}/{report['clear_pairs']}, "
                       f"{_fmt(bm['accuracy'], True)} | {_fmt(bm['log_loss'])} | {_fmt(bm['auc'])} |")
    shorter = report["shorter_winner"]
    lines = [
        "# Final test: the locked judge on the reserved test split", "",
        f"Run once, {run['started_at'][:10]} to {run['finished_at'][:10]}, on {run['device']} ({run['dtype']}). "
        f"The adapter (sha256 `{run['adapter_sha256'][:16]}…`) was chosen on validation before this run; "
        f"test file sha256 `{run['test_sha256'][:16]}…`.", "",
        "| Model | Accuracy on clear pairs | Log-loss | AUC |", "| --- | --- | --- | --- |", *rows, "",
        f"- Pairs: {report['pairs']} ({report['clear_pairs']} with a clear human preference, {report['tie_pairs']} ties, "
        f"{report['excluded_too_long']} excluded because a prompt exceeded {MAX_TOKENS} tokens; nothing was truncated).",
        f"- Both answer orders are scored and averaged. Order agreement: "
        f"{report['order_agreement'] or 0:.1%} of pairs choose the same answer in both orders.",
        f"- Pairs where the shorter answer won: {shorter.get('matches', 0)}/{shorter.get('completed', 0)} correct.",
        f"- The same adapter on validation, for comparison: {VALIDATION_RESULT}.",
        "- Intervals are 95% question-group bootstrap intervals (10,000 replicates, seed 20261007), "
        "computed with the same code as the validation reports.",
        "- The test split was not used for training, model choice or prompt design, and nothing was changed after this run.",
        "",
    ]
    if "paired_comparison" in report and report["paired_comparison"].get("status") == "completed":
        delta = report["paired_comparison"]["delta_after_minus_before"]["accuracy"]
        lines.insert(-1, f"- Accuracy gain over the unchanged model: {delta['value']:+.1%} "
                         f"(95% interval {delta['ci95'][0]:+.1%} to {delta['ci95'][1]:+.1%}).")
    return "\n".join(lines)


def load_scores(path: Path) -> dict:
    scores = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue      # a line cut off by an interruption is scored again
            scores[record["id"]] = record
        path.write_text("".join(json.dumps(r) + "\n" for r in scores.values()), encoding="utf-8")
    return scores


def main():
    parser = argparse.ArgumentParser(description="One-time final test of the locked judge on the reserved test split")
    parser.add_argument("--baseline", action="store_true", help="Also score the unchanged Qwen3-4B (doubles the time)")
    parser.add_argument("--smoke", type=int, default=0, metavar="N",
                        help="Check everything on N validation pairs instead; never reads the test split")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    adapter_sha = file_sha256(ADAPTER_DIR / "adapter_model.safetensors")
    if adapter_sha != LOCKED_ADAPTER_SHA256:
        sys.exit("The adapter in model selection/models/judge_adapter is not the locked checkpoint_0150. "
                 "The final test runs only on the judge that was chosen on validation.")
    if args.smoke:
        source, out = VALIDATION, Path(tempfile.mkdtemp(prefix="uaia_final_test_smoke_"))
        rows = read_rows(source)[: args.smoke]
    else:
        if (OUT / "report.json").exists():
            sys.exit(f"The final test has already been run; see {SUMMARY.relative_to(ROOT)}. It is not repeated.")
        source, out = TEST, OUT
        prepared = json.loads((DATA / "report.json").read_text())["output_sha256"]["test.jsonl"]
        if file_sha256(source) != prepared:
            sys.exit("test.jsonl differs from the file the data preparation produced; stopping.")
        rows = read_rows(source)
        if not rows or any(r.get("split") != "test" for r in rows):
            sys.exit("test.jsonl is empty or holds rows not marked as test; stopping.")
    out.mkdir(parents=True, exist_ok=True)

    receipt = {"source": source.name, "source_sha256": file_sha256(source), "adapter_sha256": adapter_sha,
               "rubric_sha256": RUBRIC_SHA256, "baseline": args.baseline, "dtype": args.dtype, "pairs": len(rows)}
    run_path = out / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text())
        if {key: run.get(key) for key in receipt} != receipt:
            sys.exit(f"An unfinished run with different settings is in {out}; rerun it with its original options.")
    else:
        run = receipt | {"started_at": datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
                         "calibration_opened": False}
        run_path.write_text(json.dumps(run, indent=2) + "\n")

    scores_path = out / "scores.jsonl"
    scores = load_scores(scores_path)
    todo = [r for r in rows if r["id"] not in scores]
    print(f"{len(rows)} pairs from {source.name}; {len(scores)} already scored, {len(todo)} to go.", flush=True)

    from .server import Judge
    print("Loading the judge (about a minute)...", flush=True)
    judge = Judge(BASE_DIR, ADAPTER_DIR, args.dtype)
    run["device"] = judge.device

    def tokenize(chat):
        return chat_ids(judge.tokenizer, chat)

    baseline_score = None
    if args.baseline:
        def baseline_score(ids):
            with judge.model.disable_adapter():
                return judge.score(ids)
    try:   # cross-check every prompt against the training code's own prompt builder
        sys.path.insert(0, str(ROOT / "model selection"))
        from baseline import messages_for

        def reference(row, swapped):
            return messages_for(row, swapped, RUBRIC)
    except ImportError:
        reference = None

    started = time.monotonic()
    with scores_path.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(todo, 1):
            record = pair_record(row, tokenize, judge.score, baseline_score, reference)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            scores[record["id"]] = record
            if index % 25 == 0 or index == len(todo):
                spent = time.monotonic() - started
                print(f"{len(scores)}/{len(rows)} pairs, {spent / 60:.0f} min so far, "
                      f"about {spent / index * (len(todo) - index) / 60:.0f} min left", flush=True)

    records = [scores[r["id"]] for r in rows]
    report = summarize(records, args.baseline)
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    run["test_sha256"] = run["source_sha256"]
    (out / "pairs.json").write_text(json.dumps(records, indent=1) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    markdown = summary_markdown(report, run)
    if args.smoke:
        print(markdown.replace("Final test", "Smoke check (validation pairs, not the final test)"))
        print(f"Smoke check files: {out}")
    else:
        SUMMARY.write_text(markdown, encoding="utf-8")
        print(markdown)
        print(f"Saved {SUMMARY.relative_to(ROOT)} and {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
