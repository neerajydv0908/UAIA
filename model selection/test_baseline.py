"""Check evaluation bookkeeping without loading Qwen or opening real datasets."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from baseline import classify_output, messages_for, read_rows, select_sample, summarize


def example(name, label, group=None):
    return {"id": name, "group_id": group or name, "split": "validation", "target": {"preference": label},
            "input": {"question": "Which answer helps?", "evidence": [], "answers": {"A": "short", "B": "a longer answer"}},
            "audit_only": "HIDDEN_REVIEWER_REASON"}


class BaselineTests(unittest.TestCase):
    def test_selection_is_reproducible_and_groups_are_unique(self):
        rows = [example("a", "A", "shared"), example("b", "B", "shared"), example("c", "A"), example("d", "B"), example("t", "tie_preference")]
        selected = select_sample(rows, 1)
        self.assertEqual(selected, select_sample(list(reversed(rows)), 1))
        self.assertEqual(len({row["group_id"] for row in selected}), 3)
        self.assertEqual({row["target"] for row in selected}, {"A", "B", "tie_preference"})

    def test_wrong_split_and_insufficient_groups_fail(self):
        rows = [example("a", "A"), example("b", "B"), example("t", "tie_preference")]
        for split in ("train", "calibration", "test"):
            changed = deepcopy(rows)
            changed[0]["split"] = split
            with self.assertRaises(ValueError):
                select_sample(changed, 1)
        with self.assertRaises(ValueError):
            select_sample(rows, 2)

    def test_model_input_excludes_answer_key_and_metadata(self):
        row = select_sample([example("a", "A"), example("b", "B"), example("t", "tie_preference")], 1)[0]
        row.update(target="HIDDEN_TARGET", audit_only="HIDDEN_REVIEWER_REASON")
        row["input"]["unexpected_label"] = "HIDDEN_INPUT_FIELD"
        before = deepcopy(row)
        messages = messages_for(row, True, "Judge the supplied data.")
        self.assertNotIn("HIDDEN", json.dumps(messages))
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["answers"]["A"], row["input"]["answers"]["B"])
        self.assertEqual(payload["answers"]["B"], row["input"]["answers"]["A"])
        self.assertEqual(row, before)

    def test_strict_format_and_truncation(self):
        self.assertEqual(classify_output(" A\n", "stop", True), ("completed", "B"))
        self.assertEqual(classify_output("tie_preference", "stop", True), ("completed", "tie_preference"))
        for text in ("A because it is better", "a", "B.", "", "tie"):
            self.assertEqual(classify_output(text, "stop", False), ("invalid_output", None))
        self.assertEqual(classify_output("A", "length", False), ("output_limit", None))

    def test_failures_and_missing_calls_remain_in_denominator(self):
        sample = select_sample([example("a", "A"), example("b", "B"), example("t", "tie_preference")], 1)
        records = [{"id": "a", "swapped": False, "status": "completed", "decision": "A", "seconds": 2},
                   {"id": "a", "swapped": True, "status": "completed", "decision": "A", "seconds": 4},
                   {"id": "b", "swapped": False, "status": "input_too_long", "decision": None},
                   {"id": "t", "swapped": False, "status": "invalid_output", "decision": None, "seconds": 3}]
        report = summarize(sample, records)
        self.assertEqual(report["original"]["correct"], 1)
        self.assertEqual(report["original"]["total"], 3)
        self.assertEqual(report["swapped_mapped_back"]["statuses"]["not_run"], 2)
        self.assertEqual(report["swap_consistency"], {"same_decision": 1, "valid_pairs": 1, "planned_pairs": 3})
        self.assertEqual(report["both_orders_correct"], 1)
        self.assertEqual(len(report["unresolved"]), 4)
        self.assertEqual(report["latency_seconds"], {"count": 3, "median": 3, "p95": 4})
        self.assertEqual(report["simple_rules_original"]["always_A"]["agreement"], 1 / 3)
        with self.assertRaises(ValueError):
            summarize(sample, records + records[:1])

    def test_swap_inconsistency_is_not_hidden(self):
        sample = [{"id": "a", "group_id": "a", "target": "A", "input": example("a", "A")["input"]}]
        records = [{"id": "a", "swapped": False, "status": "completed", "decision": "A"},
                   {"id": "a", "swapped": True, "status": "completed", "decision": "B"}]
        report = summarize(sample, records)
        self.assertEqual(report["swap_consistency"]["same_decision"], 0)
        self.assertEqual(report["both_orders_correct"], 0)

    def test_jsonl_keeps_unicode_line_separators_inside_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jsonl"
            row = {"text": "first\u2028second\u0085third"}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_rows(path), [row])


if __name__ == "__main__":
    unittest.main()
