"""Offline checks for fair scheduling, selection, and preserved recovery."""
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from sweep_judge import MONITORS, RATES, jobs, rank_checkpoints, recover
from train_judge import schedule_pairs


class SweepTests(unittest.TestCase):
    def records(self):
        return [{"candidate": name, "step": step, "completed": 100, "groups": 100, "log_loss": .5}
                for name in RATES for step in MONITORS]

    def test_all_rates_see_the_same_150_paired_updates(self):
        pairs = [{"id": str(i), "group_id": str(i), "orders": [{"swapped": bool(j), "tokens": [7], "target": j} for j in range(2)]} for i in range(2000)]
        batches = schedule_pairs(pairs)
        seen = []
        for name in RATES:
            parts = [s for s in jobs().values() if s["candidate"] == name]
            self.assertEqual([(s["start"], s["end"]) for s in parts], [(i, i + 25) for i in range(0, 150, 25)])
            chosen = [row for s in parts for batch in batches[s["start"]:s["end"]] for row in batch]
            self.assertEqual(len(chosen), 1200)
            self.assertEqual(len({r["id"] for r in chosen}), 600)
            for i in range(0, len(chosen), 2):
                self.assertEqual(chosen[i]["id"], chosen[i + 1]["id"])
                self.assertEqual({chosen[i]["swapped"], chosen[i + 1]["swapped"]}, {False, True})
            seen.append(chosen)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[1], seen[2])

    def test_selection_uses_loss_then_lower_rate_then_earlier_step(self):
        records = self.records()
        selected = rank_checkpoints(records)
        self.assertEqual((selected["candidate"], selected["step"]), ("lr_1e-5", 100))
        records[-1]["log_loss"] = .4
        self.assertEqual(rank_checkpoints(records), records[-1])

    def test_incomplete_duplicate_and_nonfinite_results_cannot_win(self):
        cases = [self.records()[:-1], self.records()[:-1] + [self.records()[0]]]
        for key, value in (("completed", 99), ("groups", 99), ("log_loss", math.nan), ("log_loss", math.inf), ("log_loss", -.1)):
            rows = deepcopy(self.records())
            rows[-1][key] = value
            cases.append(rows)
        for records in cases:
            with self.assertRaises(ValueError):
                rank_checkpoints(records)

    def test_recovery_preserves_failed_files_and_spent_budget(self):
        with tempfile.TemporaryDirectory() as name:
            folder, job = Path(name), "lr_1e-5_0025"
            (folder / job).mkdir()
            (folder / job / "partial.jsonl").write_text("preserve this\n")
            (folder / (job + ".log")).write_text("failure evidence\n")
            marker = folder / (job + "_worker.json")
            marker.write_text(json.dumps({"status": "worker_failed", "seconds": 17}))
            recover(folder, job)
            self.assertEqual((folder / (job + "_attempt_0001") / "partial.jsonl").read_text(), "preserve this\n")
            self.assertEqual(json.loads((folder / (job + "_attempt_0001_worker.json")).read_text())["seconds"], 17)
            self.assertFalse(marker.exists())

    def test_recovery_refuses_completed_active_or_unknown_jobs(self):
        with tempfile.TemporaryDirectory() as name:
            folder, job = Path(name), "lr_1e-5_0025"
            for state in ("completed", "running"):
                (folder / (job + "_worker.json")).write_text(json.dumps({"status": state}))
                with self.assertRaises(ValueError):
                    recover(folder, job)
            (folder / "runner.lock").mkdir()
            with self.assertRaises(ValueError):
                recover(folder, job)
            with self.assertRaises(ValueError):
                recover(folder, "not-a-segment")


if __name__ == "__main__":
    unittest.main()
