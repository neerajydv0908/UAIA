import json
import tempfile
import unittest
from pathlib import Path

from answerlab.checks import extract_claims, final_check, judge, verify
from answerlab.feedback import record_feedback, save_run
from answerlab.pipeline import run_pipeline
from answerlab.providers import FixtureProvider
from answerlab.records import Candidate, Case, FixtureReply, Limits, Passage, Run
from answerlab.retrieval import load_corpus, retrieve
from answerlab.steps import decide_sources, understand

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "examples/corpus.json"
CASE = Case.model_validate_json((ROOT / "examples/device_z.json").read_text())


class TaskTests(unittest.TestCase):
    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            understand("   ")

    def test_document_and_comparison_both_kept(self):
        task = understand("Compare these supplied manuals for their operating limits")
        self.assertEqual(set(task.kinds), {"document", "comparison"})

    def test_creative_today_does_not_automatically_search(self):
        self.assertEqual(decide_sources(understand("Write a story about today's king")).mode, "none")

    def test_current_fact_requires_web(self):
        self.assertEqual(decide_sources(understand("Who is the current president?")).mode, "fresh_web")

    def test_ambiguous_reference_needs_clarification(self):
        self.assertTrue(understand("Explain this").needs_clarification)


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.evidence = load_corpus(CORPUS)
        self.text = CASE.replies[1].text

    def test_missing_citation_remains_unknown(self):
        checks = verify(extract_claims("A plausible claim without a citation."), self.evidence)
        self.assertEqual(checks[0].status, "unknown")

    def test_uncited_extra_prose_is_not_dropped(self):
        checks = verify(extract_claims(self.text + "\nIt works underwater too."), self.evidence)
        self.assertEqual([c.status for c in checks], ["pass", "unknown"])

    def test_partial_quote_losing_condition_is_not_passed(self):
        text = "Device Z operates from 0 to 40 degrees Celsius. [device_z_operating]"
        self.assertEqual(verify(extract_claims(text), self.evidence)[0].status, "unknown")

    def test_zero_claims_do_not_mean_perfect_score(self):
        c = Candidate(id="x", provider="fixture", model="none", text="", status="completed")
        self.assertFalse(judge(c, []).eligible)

    def test_final_edit_invalidates_previous_pass(self):
        previous = verify(extract_claims(self.text), self.evidence)
        self.assertFalse(final_check(self.text.replace("40", "60"), self.evidence, previous))

    def test_source_version_change_invalidates_check(self):
        previous = verify(extract_claims(self.text), self.evidence)
        changed = [p.model_copy(update={"published_at": "2027-01-01"}) for p in self.evidence]
        self.assertFalse(final_check(self.text, changed, previous))

    def test_duplicate_source_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps([self.evidence[0].model_dump()] * 2))
            with self.assertRaises(ValueError):
                load_corpus(path)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, replies=None, question=None, **kwargs):
        replies = CASE.replies if replies is None else replies
        return await run_pipeline(question or CASE.question, [FixtureProvider(r) for r in replies], CORPUS, **kwargs)

    async def test_demo_selects_matching_quotation_and_records_all_steps(self):
        result = await self.run_case()
        self.assertEqual(result.selected_id, "sample_b")
        self.assertEqual(result.outcome, "source_quotes")
        self.assertEqual({t.step for t in result.trace}, set(range(1, 14)))
        order = [t.step for t in result.trace]
        self.assertLess(order.index(5), order.index(4))
        self.assertEqual(result.estimated_cost_usd, 0)
        self.assertEqual(result.trace[-1].status, "waiting")

    async def test_both_bad_abstains(self):
        replies = [FixtureReply(id="a", text="Wrong [device_z_operating]"), FixtureReply(id="b", text="Unsupported")]
        self.assertEqual((await self.run_case(replies)).outcome, "abstained")

    async def test_two_good_answers_report_tie(self):
        replies = [FixtureReply(id="a", text=CASE.replies[1].text), FixtureReply(id="b", text=CASE.replies[1].text)]
        self.assertEqual(set((await self.run_case(replies)).tied_ids), {"a", "b"})

    async def test_timeout_keeps_successful_sibling(self):
        replies = [FixtureReply(id="slow", text=CASE.replies[1].text, delay_seconds=0.1),
                   FixtureReply(id="fast", text=CASE.replies[1].text)]
        result = await self.run_case(replies, limits=Limits(call_timeout_seconds=0.01))
        self.assertEqual(result.selected_id, "fast")
        self.assertEqual(result.candidates[0].status, "timeout")
        self.assertTrue(result.cost_complete)

    async def test_truncated_matching_quote_is_ineligible(self):
        result = await self.run_case([FixtureReply(id="cut", text=CASE.replies[1].text, status="truncated")])
        self.assertEqual(result.outcome, "abstained")

    async def test_fresh_question_sends_no_generation_requests(self):
        result = await self.run_case(question="What is Device Z's current price?")
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.outcome, "abstained")
        self.assertIn("Fresh web", result.final_text)

    async def test_missing_evidence_abstains(self):
        result = await self.run_case(question="What does the manual say about lunar agriculture?")
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.outcome, "abstained")

    async def test_save_and_feedback_require_explicit_action(self):
        result = await self.run_case()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            feedback = Path(directory) / "feedback.jsonl"
            save_run(result, path)
            self.assertEqual(Run.model_validate_json(path.read_text()).final_hash, result.final_hash)
            with self.assertRaises(FileExistsError):
                save_run(result, path)
            with self.assertRaises(ValueError):
                record_feedback(result, feedback, "useful", "", False)
            self.assertFalse(feedback.exists())
            record = record_feedback(result, feedback, "useful", "Clear source", True, True)
            self.assertEqual(record["review_status"], "pending_human_review")
            self.assertFalse(record["training_eligible"])
            self.assertNotIn("question", record)

    async def test_real_mcp_matches_direct_retrieval(self):
        direct = await retrieve(CORPUS, CASE.question)
        through_mcp = await retrieve(CORPUS, CASE.question, "mcp")
        self.assertEqual(direct, through_mcp)


if __name__ == "__main__":
    unittest.main()
