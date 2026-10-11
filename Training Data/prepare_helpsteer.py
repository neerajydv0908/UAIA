"""Prepare local HelpSteer2 files. This script does not download or train anything."""

import argparse
import gzip
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_group(question):
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return digest(normalized)


def choose_split(official_split, group):
    if official_split == "val":
        return "validation"
    if official_split != "train":
        raise ValueError("Unrecognized publisher split")
    bucket = int(digest("helpsteer2-split-v1:" + group)[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 90 else "test"


def preference_label(score):
    if type(score) is not int or score not in range(-3, 4):
        raise ValueError("Preference strength must be an integer from -3 to 3")
    return "A" if score < 0 else "B" if score > 0 else "tie_preference"


def prepare(rows):
    official_groups = defaultdict(set)
    for row in rows:
        for key in ("prompt", "response_1", "response_2"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("Missing or empty question/answer")
        preference_label(row["preference_strength"])
        group = question_group(row["prompt"])
        choose_split(row["split"], group)
        official_groups[group].add(row["split"])
    overlap = {group for group, splits in official_groups.items() if len(splits) > 1}
    output, excluded, seen = [], [], set()
    for line, row in enumerate(rows, start=1):
        group = question_group(row["prompt"])
        reason = None
        if group in overlap:
            reason = "Question occurs in both official train and validation"
        elif row["response_1"].strip() == row["response_2"].strip():
            reason = "Identical answers; excluded from this first comparison batch"
        identity = digest(json.dumps([row[k] for k in ("prompt", "response_1", "response_2", "preference_strength")]))
        if identity in seen:
            reason = "Exact duplicate question, answers, and preference"
        seen.add(identity)
        if reason:
            excluded.append({"original_line": line, "group_id": group, "reason": reason})
            continue
        score = row["preference_strength"]
        output.append({
            "id": f"helpsteer2-preference-{line}",
            "group_id": group,
            "split": choose_split(row["split"], group),
            "task": "general_answer_preference",
            "input": {"question": row["prompt"], "evidence": [],
                      "answers": {"A": row["response_1"], "B": row["response_2"]}},
            "target": {"preference": preference_label(score), "raw_preference_strength": score},
            "grounding_label": "not_annotated",
            "review_status": "upstream_human_labels_not_locally_adjudicated",
            "provenance": {"dataset": "nvidia/HelpSteer2", "revision": REVISION,
                           "license": "CC-BY-4.0", "original_line": line, "official_split": row["split"]},
            "audit_only": {key: value for key, value in row.items()
                           if key not in ("prompt", "response_1", "response_2", "split")},
        })
    return output, excluded


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "prepared/helpsteer2_v1")
    args = parser.parse_args()
    raw = ROOT / "raw/helpsteer2"
    manifest = json.loads((raw / "download_manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected dataset revision")
    for entry in manifest["files"]:
        data = (raw / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Downloaded file changed since its manifest was written")
    with gzip.open(raw / "preference.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    prepared, excluded = prepare(rows)
    parts = {name: [r for r in prepared if r["split"] == name]
             for name in ("train", "validation", "calibration", "test")}
    group_splits = defaultdict(set)
    for row in prepared:
        group_splits[row["group_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("A question group leaked across prepared splits")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in parts.items():
        write_jsonl(args.output / (name + ".jsonl"), records)
    review = sorted(parts["train"], key=lambda row: (row["group_id"], row["id"]))[:100]
    write_jsonl(args.output / "review_first_100.jsonl", review)
    write_jsonl(args.output / "excluded.jsonl", excluded)
    report = {
        "source_revision": REVISION, "raw_rows": len(rows), "prepared_rows": len(prepared),
        "split_counts": {name: len(records) for name, records in parts.items()},
        "preference_counts": dict(Counter(r["target"]["preference"] for r in prepared)),
        "exclusions": dict(Counter(r["reason"] for r in excluded)),
        "review_rows": len(review), "cross_split_question_groups": 0,
        "split_policy": "Keep official val; hash official train groups into approximately 80/10/10 train/calibration/test",
        "limitations": ["Near-duplicate questions and shared embedded sources still need review",
                        "No evidence-grounding labels; no claim of factual correctness",
                        "Public held-out data is not a fresh product benchmark",
                        "Review file is a view of training rows, not additional training examples"],
        "training_started": False,
    }
    report["output_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(args.output.glob("*.jsonl"))}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("raw_rows", "prepared_rows", "split_counts", "exclusions", "review_rows")}, indent=2))


if __name__ == "__main__":
    main()
