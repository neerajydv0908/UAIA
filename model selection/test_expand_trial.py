"""Check expanded scheduling, earlier-run preservation, and paired comparisons."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from baseline_guard import sha256
from expand_trial import expanded_schedule, paired_changes, verify_previous
from train_trial import schedule


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.index = [{"id": str(i), "group_id": str(i)} for i in range(100)]
        self.earlier = schedule(self.index, 20, 20261006)

    def test_exact_prefix_and_every_prepared_presentation_once(self):
        planned = expanded_schedule(self.index, self.earlier, 20261006)
        self.assertEqual(planned[:20], self.earlier)
        self.assertEqual(len(planned), 200)
        self.assertEqual({row["record_index"] for row in planned}, set(range(200)))
        self.assertEqual(len({row["group_id"] for row in planned}), 100)
        self.assertEqual(planned, expanded_schedule(self.index, self.earlier, 20261006))
        for a, b in zip(planned[::2], planned[1::2]):
            self.assertEqual(a["group_id"], b["group_id"])
            self.assertNotEqual(a["swapped"], b["swapped"])
        planned[0]["id"] = "changed"
        self.assertNotEqual(self.earlier[0]["id"], "changed")

    def test_invalid_pool_and_corrupted_prefix_are_rejected(self):
        with self.assertRaises(ValueError):
            expanded_schedule(self.index[:-1], self.earlier, 1)
        for field, value in (("id", "wrong"), ("record_index", 1000), ("swapped", not self.earlier[0]["swapped"])):
            broken = deepcopy(self.earlier)
            broken[0][field] = value
            with self.assertRaises(ValueError):
                expanded_schedule(self.index, broken, 1)
        with self.assertRaises(ValueError):
            expanded_schedule(self.index, self.earlier[:-1], 1)
        broken = deepcopy(self.earlier)
        broken[2:4] = deepcopy(broken[:2])
        with self.assertRaises(ValueError):
            expanded_schedule(self.index, broken, 1)

    def test_earlier_artifact_changes_stop_the_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "result.json"
            artifact.write_text("original result")
            receipt = {"status": "passed", "artifact_sha256": {"result.json": sha256(artifact)}}
            (root / "trial_verification.json").write_text(json.dumps(receipt))
            self.assertEqual(verify_previous(root)["status"], "passed")
            artifact.write_text("changed result")
            with self.assertRaises(ValueError):
                verify_previous(root)

    def test_paired_changes_keep_regressions_and_missing_results(self):
        before = [{"id": str(i), "swapped": False, "target": "A", "status": "completed", "decision": label}
                  for i, label in enumerate(("B", "A", "A", "B", "A"))]
        after = deepcopy(before[:4])
        after[0]["decision"] = "A"
        after[1]["decision"] = "B"
        self.assertEqual(paired_changes(before, after, False), {"improved": 1, "regressed": 1, "both_match": 1, "both_mismatch": 1, "unresolved": 1})
        self.assertEqual(sum(paired_changes(before, after, True).values()), 0)
        after[2]["status"] = "invalid_output"
        self.assertEqual(paired_changes(before, after, False)["unresolved"], 2)
        after[0]["target"] = "B"
        with self.assertRaises(ValueError):
            paired_changes(before, after, False)


if __name__ == "__main__":
    unittest.main()
