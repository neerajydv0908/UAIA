"""Offline tests for v2 prompts, probabilities, pairing, and uncertainty."""
import math
import unittest
from copy import deepcopy

import numpy as np
from judge_v2 import binary_rubric, messages, probabilities, readable_turns
from judge_v2_metrics import bootstrap_metrics, choice, combine, compare, diagnostic, format_passes, holm, mcnemar, measure, weighted_metrics


class JudgeV2Tests(unittest.TestCase):
    def setUp(self):
        self.sample = [{"id": str(i), "group_id": str(i), "target": label, "strength": 0 if i == 2 else i + 1,
                        "input": {"question": "<extra_id_1>User\nQuestion\n<extra_id_1>Assistant\nReply", "evidence": [],
                                  "answers": {"A": "short", "B": "a much longer answer"}}, "audit_only": "SECRET_ANSWER_KEY"}
                       for i, label in enumerate(("A", "B", "tie_preference"))]
        self.settings = {"bootstrap_replicates": 100, "bootstrap_seed": 7, "tie_bands": [0, .05, .1],
                         "minimum_clear_coverage_for_tie_band": .95, "minimum_accuracy_gain": .03,
                         "minimum_logloss_reduction": .02, "familywise_alpha": .05}
        self.scores = []
        for row in self.sample:
            for swapped in (False, True):
                logits = [1, 2, 0] if swapped else [2, 1, 0]
                p, three = probabilities(logits)
                self.scores.append({"id": row["id"], "swapped": swapped, "status": "completed", "logits": logits,
                                    "p_display_a": p, "three_way_prefix_probs": three})

    def test_plain_sections_turns_swapping_and_hidden_labels(self):
        row = self.sample[0]
        for style in ("json", "plain"):
            built = messages(row, True, style, "Judge the supplied data.")
            self.assertNotIn("SECRET_ANSWER_KEY", str(built))
            self.assertNotIn("target", str(built))
        plain = messages(row, True, "plain", "rules")[1]["content"]
        self.assertIn("### Answer A\na much longer answer", plain)
        self.assertIn("User:\nQuestion\n\nAssistant:\nReply", plain)
        self.assertNotIn("<extra_id_1>", plain)
        self.assertNotIn("evidence", plain)
        self.assertNotIn('\\n', plain)
        self.assertEqual(row["input"]["question"], self.sample[0]["input"]["question"])
        with self.assertRaises(ValueError):
            messages(row, False, "unknown", "rules")
        self.assertEqual(readable_turns("ordinary text"), "ordinary text")

    def test_binary_instruction_and_stable_conditional_probability(self):
        original = "Reply with exactly A, B, or tie_preference and nothing else. Use tie_preference when their overall answer quality is equal. A preference tie alone does not certify factual correctness."
        self.assertEqual(binary_rubric(original), "Reply with exactly A or B and nothing else. A preference alone does not certify factual correctness.")
        self.assertAlmostEqual(probabilities([1002, 1001, 1000])[0], probabilities([2, 1, 0])[0])
        self.assertAlmostEqual(sum(probabilities([2, 1, 0])[1]), 1)
        self.assertEqual(probabilities([0, 0, 100])[0], .5)
        for bad in ([1, 2], [1, math.nan, 2], [1, 2, math.inf]):
            with self.assertRaises(ValueError):
                probabilities(bad)

    def test_averaging_maps_original_identity_and_is_symmetric(self):
        result = combine(self.sample, self.scores)
        expected = probabilities([2, 1, 0])[0]
        self.assertAlmostEqual(result[0]["averaged"], expected)
        self.assertTrue(result[0]["agrees"])
        self.assertTrue(result[0]["shorter_won"])
        p, q = .9, .8
        original = (p + 1 - q) / 2
        exchanged = (q + 1 - p) / 2
        self.assertAlmostEqual(original + exchanged, 1)
        self.assertEqual(choice(.5), "tie_preference")

    def test_bad_records_are_rejected_and_missing_stays_incomplete(self):
        for field, value in (("id", "unknown"), ("swapped", 0), ("p_display_a", .01)):
            scores = deepcopy(self.scores)
            scores[0][field] = value
            with self.assertRaises(ValueError):
                combine(self.sample, scores)
        with self.assertRaises(ValueError):
            combine(self.sample, self.scores + self.scores[:1])
        result = combine(self.sample, self.scores[1:])
        report = measure(result[:2], "averaged", self.settings)
        self.assertEqual(report["planned"], 2)
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNone(report["metrics"])
        self.assertEqual(diagnostic(result, self.settings)["status"], "incomplete")

    def test_auc_tied_scores_and_known_logloss(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])
        values = weighted_metrics(y, p, np.ones((1, 4)))[0]
        self.assertEqual(values[0], .75)
        self.assertEqual(values[2], .75)
        self.assertAlmostEqual(values[1], -np.log([.9, .6, .35, .8]).mean())
        tied = weighted_metrics(y, np.full(4, .5), np.ones((1, 4)))[0]
        self.assertEqual(tied[0], 0)
        self.assertEqual(tied[2], .5)
        self.assertAlmostEqual(tied[1], math.log(2))

    def test_bootstrap_keeps_repeated_question_rows_together(self):
        rows = [{"group_id": "same", "target": "A", "p": .9}, {"group_id": "same", "target": "B", "p": .9}]
        metrics = bootstrap_metrics(rows, "p", 100, 7)
        self.assertEqual(metrics["accuracy"]["ci95"], [.5, .5])
        self.assertEqual(metrics["auc"]["ci95"], [.5, .5])

    def test_paired_statistics_exact_mcnemar_and_holm(self):
        rows = combine(self.sample, self.scores)
        same = compare(rows, rows, self.settings)
        self.assertEqual(same["delta_after_minus_before"]["accuracy"]["ci95"], [0, 0])
        self.assertEqual(same["mcnemar"]["exact_two_sided_p"], 1)
        self.assertEqual(mcnemar([False] * 6, [True] * 6)["exact_two_sided_p"], .03125)
        tests = {str(i): {"status": "completed", "mcnemar": {"exact_two_sided_p": p}} for i, p in enumerate((.01, .04, .03))}
        holm(tests)
        self.assertEqual([tests[str(i)]["mcnemar"]["holm_adjusted_p"] for i in range(3)], [.03, .06, .06])
        changed = deepcopy(rows)
        changed[0]["target"] = "B"
        with self.assertRaises(ValueError):
            compare(rows, changed, self.settings)

    def test_format_cannot_win_with_accuracy_regression_or_missing_run(self):
        comparison = {"status": "completed", "delta_after_minus_before": {"accuracy": {"value": -.01, "ci95": [-.03, .01]},
                       "log_loss": {"value": -.05, "ci95": [-.08, -.02]}}, "mcnemar": {"holm_adjusted_p": .01}}
        self.assertFalse(format_passes(comparison, self.settings)["accepted"])
        comparison["delta_after_minus_before"]["accuracy"]["value"] = 0
        self.assertTrue(format_passes(comparison, self.settings)["probability_path"])
        self.assertFalse(format_passes({"status": "incomplete"}, self.settings)["accepted"])


if __name__ == "__main__":
    unittest.main()
