"""Test bounded scheduling, readiness checks, and interrupted-result handling."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baseline_guard import sha256
from train_trial import check_ready, collect_predictions, schedule, supervise


class TrialTests(unittest.TestCase):
    def test_schedule_pairs_orders_and_is_repeatable(self):
        index = [{"id": str(i), "group_id": str(i)} for i in range(100)]
        planned = schedule(index, 20, 123)
        self.assertEqual(planned, schedule(index, 20, 123))
        self.assertEqual(len(planned), 20)
        self.assertEqual(len({row["group_id"] for row in planned}), 10)
        for first, second in zip(planned[::2], planned[1::2]):
            self.assertEqual(first["group_id"], second["group_id"])
            self.assertEqual({first["swapped"], second["swapped"]}, {False, True})
            for row in (first, second):
                self.assertEqual(row["record_index"], 2 * int(row["id"]) + int(row["swapped"]))

    def test_bad_schedule_size_or_duplicate_groups_fails(self):
        index = [{"id": str(i), "group_id": str(i)} for i in range(10)]
        for count in (0, -2, 3, 22):
            with self.assertRaises(ValueError):
                schedule(index, count, 123)
        index[1]["group_id"] = index[0]["group_id"]
        with self.assertRaises(ValueError):
            schedule(index, 20, 123)

    def test_readiness_rejects_changed_files_or_unpassed_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            config = {"batch_size": 1, "max_seq_length": 2048, "lora_layers": 4, "mask_prompt": True,
                      "grad_checkpoint": True, "optimizer": "Adam", "future_trial_iterations": 20,
                      "future_trial_wall_seconds": 900, "learning_rate": 0.00001}
            (path / "training_config.json").write_text(json.dumps(config))
            (path / "profile.json").write_text(json.dumps({"files_sha256": {"training_config.json": sha256(path / "training_config.json")}}))
            probe = path / "probe_result.json"
            probe.write_text(json.dumps({"status": "passed", "optimizer_updates": 0}))
            self.assertEqual(check_ready(path)[0], config)
            probe.write_text(json.dumps({"status": "failed", "optimizer_updates": 0}))
            with self.assertRaises(ValueError):
                check_ready(path)
            probe.write_text(json.dumps({"status": "passed", "optimizer_updates": 0}))
            (path / "training_config.json").write_text("changed")
            with self.assertRaises(ValueError):
                check_ready(path)

    def test_failed_and_timed_out_workers_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch("train_trial.subprocess.run", return_value=SimpleNamespace(returncode=1)) as run:
                result = supervise(path, "train", 900)
                self.assertEqual(result["status"], "worker_failed")
                self.assertEqual(run.call_args.kwargs["timeout"], 900)
            with patch("train_trial.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 900)):
                result = supervise(path, "evaluate", 900)
                self.assertEqual(result["status"], "deadline_exceeded")
            saved = json.loads((path / "evaluate_worker.json").read_text())
            self.assertEqual(saved["status"], "deadline_exceeded")

    def test_missing_and_partial_prediction_files_do_not_look_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            self.assertEqual(collect_predictions(path), ([], False))
            path.write_text('{"id": "finished"}\n{"id": "interrupted')
            self.assertEqual(collect_predictions(path), ([{"id": "finished"}], True))


if __name__ == "__main__":
    unittest.main()
