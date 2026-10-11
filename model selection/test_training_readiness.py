"""Check the baseline guard and training formatting without using the GPU."""
import json
import tempfile
import unittest
from pathlib import Path

from baseline_guard import sha256, verify
from prepare_training import length_summary, tokenize_record, training_record


class TrainingReadinessTests(unittest.TestCase):
    def row(self, split="train"):
        return {"split": split, "target": {"preference": "A"}, "audit_only": "DO_NOT_SHOW_REVIEW",
                "input": {"question": "Which?", "evidence": [], "answers": {"A": "First", "B": "Second"}}}

    def test_swap_changes_answer_contents_and_training_label(self):
        original = training_record(self.row(), False, "Judge.")
        swapped = training_record(self.row(), True, "Judge.")
        self.assertEqual(original["messages"][-1]["content"], "A")
        self.assertEqual(swapped["messages"][-1]["content"], "B")
        self.assertEqual(json.loads(swapped["messages"][1]["content"])["answers"]["A"], "Second")
        self.assertNotIn("DO_NOT_SHOW_REVIEW", json.dumps(swapped))
        self.assertNotIn('"target"', swapped["messages"][1]["content"])

    def test_ties_remain_ties_and_wrong_splits_fail(self):
        row = self.row()
        row["target"]["preference"] = "tie_preference"
        self.assertEqual(training_record(row, True, "Judge.")["messages"][-1]["content"], "tie_preference")
        for split in ("validation", "calibration", "test"):
            with self.assertRaises(ValueError):
                training_record(self.row(split), False, "Judge.")

    def test_prompt_prefix_must_match_complete_sequence(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
                if return_dict:
                    raise AssertionError("Require token-ID lists, not a dictionary.")
                return [1, 2] if add_generation_prompt else [1, 2, 3, 4]
        class BrokenTokenizer(Tokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
                return [1, 2] if add_generation_prompt else [1, 5, 3, 4]
        record = training_record(self.row(), False, "Judge.")
        self.assertEqual(tokenize_record(record, Tokenizer()), ([1, 2, 3, 4], 2))
        with self.assertRaises(ValueError):
            tokenize_record(record, BrokenTokenizer())

    def test_length_percentiles_use_nearest_rank(self):
        self.assertEqual(length_summary([4, 1, 3, 2]), {"count": 4, "min": 1, "max": 4, "p50": 2, "p90": 4, "p95": 4, "p99": 4})

    def test_guard_rejects_changed_missing_and_writable_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "fixed.txt"
            path.write_text("baseline")
            lock = root / "lock.json"
            lock.write_text(json.dumps({"baseline_id": "test", "mac_immutable": False, "files": {"fixed.txt": sha256(path)}}))
            with self.assertRaises(ValueError):
                verify(root, Path("lock.json"))
            path.chmod(0o444)
            self.assertEqual(verify(root, Path("lock.json"))["status"], "passed")
            path.chmod(0o644)
            path.write_text("changed")
            path.chmod(0o444)
            with self.assertRaises(ValueError):
                verify(root, Path("lock.json"))
            path.unlink()
            with self.assertRaises(ValueError):
                verify(root, Path("lock.json"))


if __name__ == "__main__":
    unittest.main()
