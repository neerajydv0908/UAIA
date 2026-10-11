"""Small offline checks for label direction and separation of practice/exam data."""

import unittest

from prepare_helpsteer import choose_split, preference_label, prepare, question_group


def example(prompt="Question", split="train", score=1):
    return {"prompt": prompt, "split": split, "response_1": "Answer A",
            "response_2": "Answer B", "preference_strength": score}


class PreparationTests(unittest.TestCase):
    def test_negative_means_first_answer_wins(self):
        self.assertEqual(preference_label(-3), "A")
        self.assertEqual(preference_label(3), "B")
        self.assertEqual(preference_label(0), "tie_preference")

    def test_invalid_score_is_rejected(self):
        for score in (True, 4, "1", None):
            with self.assertRaises(ValueError):
                preference_label(score)

    def test_question_case_and_spacing_share_a_group(self):
        self.assertEqual(question_group(" What   NOW? "), question_group("what now?"))

    def test_official_validation_never_becomes_training(self):
        self.assertEqual(choose_split("val", question_group("Question")), "validation")

    def test_cross_official_split_question_is_quarantined(self):
        kept, excluded = prepare([example(), example(" QUESTION ", "val")])
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 2)

    def test_missing_evidence_is_not_invented(self):
        kept, _ = prepare([example()])
        self.assertEqual(kept[0]["input"]["evidence"], [])
        self.assertEqual(kept[0]["grounding_label"], "not_annotated")

    def test_duplicates_and_identical_answers_are_excluded(self):
        identical = example("Another question")
        identical["response_2"] = identical["response_1"]
        kept, excluded = prepare([example(), example(), identical])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(excluded), 2)

    def test_same_question_candidates_stay_together(self):
        kept, _ = prepare([example(score=-1), example(score=2)])
        self.assertEqual(kept[0]["split"], kept[1]["split"])

    def test_unknown_split_and_empty_answers_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare([example(split="mystery")])
        row = example()
        row["response_1"] = " "
        with self.assertRaises(ValueError):
            prepare([row])


if __name__ == "__main__":
    unittest.main()
