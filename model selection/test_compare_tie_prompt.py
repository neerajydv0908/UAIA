"""Offline checks for the fixed-weight prompt experiment; no model calls."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from baseline_guard import ROOT, SNAPSHOT, sha256
from compare_tie_prompt import MODELS, REVIEW, acceptance_screen, checked_summary, text_hash, validate_proposal, verify_receipt


class PromptComparisonTests(unittest.TestCase):
    def setUp(self):
        self.proposal = json.loads((REVIEW / "next_experiment_plan.json").read_text())
        self.original = json.loads((ROOT / SNAPSHOT / "plan.json").read_text())["rubric"]
        self.sample = [{"id": str(i), "target": ("A", "B", "tie_preference")[i // 20],
                        "input": {"answers": {"A": "one", "B": "two"}}} for i in range(60)]
        perfect = [{"id": row["id"], "target": row["target"], "swapped": swapped,
                    "status": "completed", "decision": row["target"]} for row in self.sample for swapped in (False, True)]
        self.records = {name: deepcopy(perfect) for name in MODELS}

    def screen(self):
        metrics = {name: checked_summary(self.sample, rows) for name, rows in self.records.items()}
        return acceptance_screen(metrics, self.records, self.proposal["development_acceptance_screen"])

    def test_exact_proposal_is_accepted(self):
        validate_proposal(self.proposal, self.original)
        self.assertTrue(self.screen()["passed"])

    def test_extra_prompt_change_wrong_hash_and_larger_budget_are_rejected(self):
        for field, value in (("candidate_rubric", self.proposal["candidate_rubric"] + " Extra rule."),
                             ("candidate_rubric_sha256", "wrong"), ("planned_new_decisions", 242),
                             ("calibration_or_test_opened", True), ("new_training_updates", 1)):
            proposal = deepcopy(self.proposal)
            proposal[field] = value
            if field == "candidate_rubric":
                proposal["candidate_rubric_sha256"] = text_hash(value)
            with self.assertRaises(ValueError):
                validate_proposal(proposal, self.original)

    def test_duplicate_unknown_id_wrong_label_and_non_boolean_order_are_rejected(self):
        original = self.records["baseline"]
        with self.assertRaises(ValueError):
            checked_summary(self.sample, original + original[:1])
        for field, value in (("id", "unknown"), ("target", "B"), ("swapped", 0)):
            rows = deepcopy(original)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                checked_summary(self.sample, rows)

    def test_missing_and_invalid_decisions_cannot_pass(self):
        for name in MODELS:
            with self.subTest(model=name):
                row = self.records[name].pop()
                metrics = checked_summary(self.sample, self.records[name])
                self.assertEqual(metrics["swapped_mapped_back"]["total"], 60)
                self.assertEqual(metrics["unresolved"][0]["status"], "not_run")
                self.assertFalse(self.screen()["all_240_planned_decisions_valid"])
                self.records[name].append({**row, "status": "invalid_output", "decision": None})
                self.assertFalse(self.screen()["passed"])
                self.records[name][-1] = row

    def test_more_ties_cannot_hide_lost_clear_winners(self):
        for row in self.records["updates200"]:
            row["decision"] = "tie_preference"
        screen = self.screen()
        self.assertTrue(screen["all_240_planned_decisions_valid"])
        self.assertTrue(screen["thresholds"]["original_tie_matches_min"]["passed"])
        self.assertFalse(screen["thresholds"]["original_clear_matches_min"]["passed"])
        self.assertFalse(screen["passed"])

    def test_earlier_artifacts_and_failed_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "result.json"
            artifact.write_text("original result")
            receipt = {"status": "passed", "artifact_sha256": {"result.json": sha256(artifact)}}
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(verify_receipt(root, path.name)["files_checked"], 1)
            artifact.write_text("changed result")
            with self.assertRaises(ValueError):
                verify_receipt(root, path.name)
            receipt.update(status="failed", artifact_sha256={})
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                verify_receipt(root, path.name)


if __name__ == "__main__":
    unittest.main()
