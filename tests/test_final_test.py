import importlib.util
import json
import unittest

from judge_server.final_test import ROOT, pair_record, summarize, summary_markdown

ANSWER_A, ANSWER_B = "A short answer.", "A much longer answer with more words."


def row(label="A", strength=2, question="Which is better?", pair_id="p1", group="g1"):
    return {"id": pair_id, "group_id": group, "split": "test",
            "input": {"question": question, "evidence": [], "answers": {"A": ANSWER_A, "B": ANSWER_B}},
            "target": {"preference": label, "raw_preference_strength": -strength}}


def tokenize(chat):
    shown_first = json.loads(chat[1]["content"])["answers"]["A"]
    return [1 if shown_first == ANSWER_A else 2] * 10


def score(ids):          # P(shown A is better): 0.8 in the original order, 0.3 when swapped
    return 0.8 if ids[0] == 1 else 0.3


class PairRecordTests(unittest.TestCase):
    def test_both_orders_are_mapped_back_and_averaged(self):
        record = pair_record(row(), tokenize, score)
        self.assertEqual(record["status"], "completed")
        self.assertAlmostEqual(record["original"], 0.8)
        self.assertAlmostEqual(record["swapped"], 0.7)        # 1 - P(B shown first is better)
        self.assertAlmostEqual(record["p_a"], 0.75)
        self.assertEqual(record["strength"], 2)
        self.assertTrue(record["shorter_won"])                 # A is shorter and won
        self.assertNotIn("input", record)
        self.assertNotIn(ANSWER_A, json.dumps(record))          # no text is stored

    def test_baseline_is_scored_on_the_same_prompts(self):
        record = pair_record(row(), tokenize, score, baseline_score=lambda ids: 0.5)
        self.assertAlmostEqual(record["baseline_p_a"], 0.5)

    def test_long_prompts_are_excluded_not_truncated(self):
        record = pair_record(row(), lambda chat: [1] * 5000, score)
        self.assertEqual(record["status"], "too_long")
        self.assertNotIn("p_a", record)

    def test_prompt_must_match_the_training_format(self):
        with self.assertRaises(ValueError):
            pair_record(row(), tokenize, score, reference=lambda r, swapped: [])

    def test_bad_probability_stops_the_run(self):
        with self.assertRaises(RuntimeError):
            pair_record(row(), tokenize, lambda ids: float("nan"))


@unittest.skipUnless((ROOT / "model selection/baseline.py").exists(), "needs the training prompt builder")
class TrainingFormatTests(unittest.TestCase):
    def test_served_prompts_equal_the_training_prompts(self):
        import sys
        sys.path.insert(0, str(ROOT / "model selection"))
        from baseline import messages_for

        from judge_server.server import RUBRIC
        sample = row(question="Café: “quoted” <extra_id_1>User\nnext turn")
        sample["input"]["evidence"] = [{"id": "e1", "text": "naïve"}]
        captured = []
        pair_record(sample, lambda chat: captured.append(chat) or [1], lambda ids: 0.5,
                    reference=lambda r, swapped: messages_for(r, swapped, RUBRIC))
        self.assertEqual(captured, [messages_for(sample, s, RUBRIC) for s in (False, True)])


@unittest.skipUnless(importlib.util.find_spec("numpy") and (ROOT / "model selection/colab/metrics.py").exists(),
                     "needs numpy and the validation metric code")
class SummaryTests(unittest.TestCase):
    def test_summary_uses_clear_pairs_and_reports_exclusions(self):
        rows = [row("A", pair_id="p1", group="g1"), row("B", pair_id="p2", group="g2"),
                row("tie_preference", pair_id="p3", group="g3"), row("A", pair_id="p4", group="g4")]
        records = [pair_record(r, tokenize, score, baseline_score=lambda ids: 0.5) for r in rows[:3]]
        records.append(pair_record(rows[3], lambda chat: [1] * 5000, score))
        report = summarize(records, with_baseline=True)
        self.assertEqual((report["pairs"], report["clear_pairs"], report["tie_pairs"], report["excluded_too_long"]), (4, 2, 1, 1))
        self.assertEqual(report["judge"]["averaged"]["matches"], 1)       # p_a = 0.75 is right for A, wrong for B
        run = {"started_at": "2026-10-08T00:00:00", "finished_at": "2026-10-08T01:00:00", "device": "mps",
               "dtype": "bfloat16", "adapter_sha256": "0" * 64, "test_sha256": "1" * 64}
        text = summary_markdown(report, run)
        self.assertIn("1/2", text)
        self.assertIn("Unchanged Qwen3-4B", text)
        self.assertIn("1 excluded", text)


if __name__ == "__main__":
    unittest.main()
