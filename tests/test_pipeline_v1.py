import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from answerlab.judge_client import JudgeUnavailable, judge_pair
from answerlab.pipeline_v1 import (MAX_PROMPT_BYTES, build_live, fit_evidence, recheck, run_v1, spending_plan,
                                   strip_citations)
from answerlab.providers import HTTPProvider, make_prompt
from answerlab.claims_cited import extract_cited_claims
from answerlab.records import Candidate, LiveConfig, Limits, Passage, ProviderConfig
from answerlab.verify_semantic import verify_semantic
from answerlab.web_search import SearchNotSent

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "examples/corpus.json"
WEB = [Passage(id="web_1", title="Boiling point", location="https://example.org/boil",
               text="At sea level, water boils at 100 degrees Celsius."),
       Passage(id="web_2", title="Altitude", location="https://example.org/altitude",
               text="At higher altitudes, water boils at lower temperatures because air pressure is lower.")]
GOOD_1 = "Water boils at 100 degrees Celsius at sea level. [web_1]"
GOOD_2 = ("At sea level, water boils at 100 degrees Celsius. [web_1] "
          "It boils at lower temperatures at higher altitudes. [web_2]")
DEVICE_Z = "According to the manual, what temperatures does Device Z operate in?"
DEVICE_Z_ANSWER = "Device Z operates from 0 to 40 degrees Celsius in dry conditions. [device_z_operating]"
UNCERTAIN = "What temperatures does Device Z operate in?"   # saved matches plus a web search
LIMITS = Limits(max_estimated_usd=1, request_timeout_seconds=30)


class FakeModel:
    def __init__(self, id, text, status="completed"):
        self.id, self.provider, self.model = id, "fake", id
        self.text, self.status = text, status
        self.instructions = None

    async def generate(self, prompt, limits, instructions=None):
        self.instructions = instructions
        return Candidate(id="ignored", provider="fake", model=self.model, text=self.text,
                         status=self.status, estimated_cost_usd=0.001)


class FakeVerifier:
    """Labels a claim SUPPORTED unless it contains a word listed in `contradict` or `unsure`."""
    model = "fake-verifier"

    def __init__(self, contradict=(), unsure=()):
        self.contradict, self.unsure, self.calls = contradict, unsure, 0

    async def complete(self, system, prompt, limits):
        self.calls += 1
        items = json.loads(prompt)["claims"]
        def label(text):
            if any(word in text for word in self.contradict):
                return "CONTRADICTED"
            if any(word in text for word in self.unsure):
                return "NOT_ENOUGH_INFO"
            return "SUPPORTED"
        return json.dumps({"checks": [{"index": i["index"], "label": label(i["claim"]), "reason": "fake"}
                                      for i in items]}), 0.0005


async def search(question):
    return list(WEB)


def judge_prefers_b(calls):
    async def judge(question, a, b):
        calls.append((a, b))
        return {"p_a": 0.2, "p_a_original_order": 0.25, "p_a_swapped_order": 0.15, "seconds": 0.1}
    return judge


async def judge_down(question, a, b):
    raise JudgeUnavailable("Judge server not reachable")


async def judge_even(question, a, b):
    return {"p_a": 0.5, "p_a_original_order": 0.5, "p_a_swapped_order": 0.5, "seconds": 0.1}


def paid_provider(client):
    config = ProviderConfig(provider="openai", model="test", input_usd_per_million=1, output_usd_per_million=2)
    provider = HTTPProvider(config, client)
    provider.id = "openai:test"
    return provider


class PipelineV1Tests(unittest.IsolatedAsyncioTestCase):
    async def ask(self, question="How hot does water need to be to boil?", models=None, verifier=None, **kwargs):
        models = models if models is not None else [FakeModel("m1", GOOD_1), FakeModel("m2", GOOD_2)]
        kwargs.setdefault("search_fn", search)
        kwargs.setdefault("claims_fn", extract_cited_claims)
        kwargs.setdefault("verify_fn", verify_semantic)
        return await run_v1(question, models, verifier or FakeVerifier(), CORPUS, LIMITS, mode="fixture", **kwargs)

    async def test_trained_judge_picks_between_verified_answers(self):
        calls = []
        models = [FakeModel("m1", GOOD_1), FakeModel("m2", GOOD_2)]
        run = await self.ask(models=models, judge_fn=judge_prefers_b(calls))
        self.assertEqual(run.outcome, "verified_answer")
        self.assertEqual(run.selected_id, "m2")
        self.assertTrue(run.judge.used)
        self.assertEqual(len(run.judge.comparisons), 1)
        self.assertNotIn("[web_1]", calls[0][0])           # the judge sees plain answers
        self.assertEqual({t.step for t in run.trace}, set(range(1, 14)))
        order = [t.step for t in run.trace]
        self.assertLess(order.index(5), order.index(4))
        self.assertIn("[id]", models[0].instructions)      # cited-answer instructions were sent
        self.assertGreater(run.estimated_cost_usd, 0)

    async def test_judge_down_falls_back_to_rules(self):
        run = await self.ask(judge_fn=judge_down)
        self.assertEqual(run.outcome, "verified_answer")
        self.assertFalse(run.judge.used)
        self.assertIn("rule-based", run.judge.reason)

    async def test_contradicted_answer_is_never_selected(self):
        models = [FakeModel("m1", "Water boils at 50 degrees Celsius at sea level. [web_1]"), FakeModel("m2", GOOD_2)]
        run = await self.ask(models=models, verifier=FakeVerifier(contradict=("50 degrees",)), judge_fn=judge_down)
        self.assertEqual(run.selected_id, "m2")
        self.assertIn("Only one answer", run.judge.reason)
        self.assertFalse(next(j for j in run.judgments if j.candidate_id == "m1").eligible)

    async def test_all_unsupported_abstains(self):
        run = await self.ask(verifier=FakeVerifier(contradict=("boil",)), judge_fn=judge_down)
        self.assertEqual(run.outcome, "abstained")
        self.assertIsNone(run.selected_id)

    async def test_mostly_supported_answer_is_partly_verified(self):
        text = " ".join(f"Fact {i} about boiling water at sea level. [web_1]" for i in range(4))
        text += " A vague extra remark. [web_2]"
        run = await self.ask(models=[FakeModel("m1", text)], verifier=FakeVerifier(unsure=("vague",)))
        self.assertEqual(run.outcome, "partly_verified")
        self.assertIn("4 of 5", run.uncertainty)

    async def test_uncited_answer_is_not_eligible(self):
        run = await self.ask(models=[FakeModel("m1", "Water boils at 100 degrees Celsius.")])
        self.assertEqual(run.outcome, "abstained")

    async def test_fresh_question_without_search_abstains(self):
        async def broken(question):
            raise ValueError("Set TAVILY_API_KEY in your environment before web search")
        run = await self.ask("What is the current price of gold?", search_fn=broken)
        self.assertEqual(run.outcome, "abstained")
        self.assertIn("TAVILY_API_KEY", run.final_text)
        self.assertEqual(run.candidates, [])

    async def test_saved_document_question_skips_web(self):
        called = []
        async def tracking(question):
            called.append(question)
            return []
        answer = "Device Z operates from 0 to 40 degrees Celsius in dry conditions. [device_z_operating]"
        run = await self.ask("According to the manual, what temperatures does Device Z operate in?",
                             models=[FakeModel("m1", answer)], search_fn=tracking)
        self.assertEqual(called, [])
        self.assertEqual(run.outcome, "verified_answer")
        self.assertIn("fictional", run.uncertainty)

    async def test_no_models_explains_keys(self):
        run = await self.ask(models=[])
        self.assertEqual(run.outcome, "abstained")
        self.assertIn("API_KEY", run.final_text)

    async def test_budget_blocks_before_any_http(self):
        calls = []
        async def handler(request):
            calls.append(request)
            return httpx.Response(200, json={})
        config = ProviderConfig(provider="openai", model="test", input_usd_per_million=1, output_usd_per_million=2)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                provider = HTTPProvider(config, client)
                run = await run_v1("How hot does water need to be to boil?", [provider], provider, CORPUS,
                                   Limits(max_estimated_usd=0.0001), search_fn=search,
                                   claims_fn=extract_cited_claims, verify_fn=verify_semantic)
        self.assertEqual(calls, [])
        self.assertEqual(run.outcome, "abstained")
        self.assertIn("budget", run.final_text)

    async def test_zero_budget_sends_no_web_search(self):
        # Review case: the search ran before any spending check, then the run reported zero cost.
        searched, calls = [], []
        async def tracking(question):
            searched.append(question)
            return list(WEB)
        async def handler(request):
            calls.append(request)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                provider = paid_provider(client)
                run = await run_v1("What is the current price of gold?", [provider], provider, CORPUS,
                                   Limits(max_estimated_usd=0), search_fn=tracking, search_usd=0.008,
                                   claims_fn=extract_cited_claims, verify_fn=verify_semantic)
        self.assertEqual((searched, calls), ([], []))
        self.assertEqual(run.outcome, "abstained")
        self.assertIn("nothing was sent", run.final_text)
        self.assertEqual((run.estimated_cost_usd, run.cost_complete), (0, True))

    async def test_web_search_cost_is_counted(self):
        run = await self.ask("What is the current price of gold?", judge_fn=judge_down, search_usd=0.008)
        # Two answers at 0.001, two verifier calls at 0.0005, plus one search.
        self.assertAlmostEqual(run.estimated_cost_usd, 0.002 + 0.001 + 0.008)
        self.assertTrue(run.cost_complete)
        self.assertIn("web search about $0.0080", next(t.detail for t in run.trace if t.step == 5))

    async def test_failed_paid_search_leaves_cost_unknown(self):
        async def broken(question):
            raise httpx.ConnectError("search service down")
        run = await self.ask("What is the current price of gold?", search_fn=broken, search_usd=0.008)
        self.assertEqual(run.outcome, "abstained")
        self.assertFalse(run.cost_complete)

    async def test_search_refused_before_sending_costs_nothing(self):
        # A search that is refused before sending (here: no search key) must not make the cost "unknown".
        async def no_key(question):
            raise SearchNotSent("Set TAVILY_API_KEY in your environment before web search")
        run = await self.ask(UNCERTAIN, models=[FakeModel("m1", DEVICE_Z_ANSWER)], search_fn=no_key, search_usd=0.008)
        self.assertEqual(run.outcome, "verified_answer")
        self.assertTrue(run.cost_complete)
        self.assertIn("web search skipped (Set TAVILY_API_KEY", next(t.detail for t in run.trace if t.step == 5))

    async def test_failed_paid_search_still_counts_against_the_budget(self):
        shares = []
        async def broken(question):
            raise httpx.ReadTimeout("search timed out after sending")
        async def recording_verify(claims, evidence, verifier, limits):
            shares.append(limits.max_estimated_usd)
            return await verify_semantic(claims, evidence, verifier, limits)
        run = await self.ask(UNCERTAIN, models=[FakeModel("m1", DEVICE_Z_ANSWER)], search_fn=broken,
                             search_usd=0.008, verify_fn=recording_verify)
        self.assertEqual(run.outcome, "verified_answer")      # the saved passage still answers it
        self.assertAlmostEqual(shares[0], 1 - 0.008 - 0.001)   # budget minus the search and the answer
        self.assertFalse(run.cost_complete)
        self.assertIn("may still have been billed", next(t.detail for t in run.trace if t.step == 5))

    async def test_search_cancelled_by_the_deadline_leaves_cost_unknown(self):
        async def slow(question):
            await asyncio.sleep(5)
            return list(WEB)
        run = await run_v1("What is the current price of gold?", [FakeModel("m1", GOOD_1)], FakeVerifier(), CORPUS,
                           Limits(max_estimated_usd=1, request_timeout_seconds=0.5), search_fn=slow, search_usd=0.008,
                           claims_fn=extract_cited_claims, verify_fn=verify_semantic, mode="fixture")
        self.assertEqual(run.outcome, "abstained")
        self.assertFalse(run.cost_complete)

    async def test_search_needs_budget_for_the_largest_follow_up(self):
        # Priced at the smallest possible prompt, a budget could pay for the search and then refuse every
        # answer call. The check now prices the follow-up calls at the largest prompt allowed.
        searched = []
        async def tracking(question):
            searched.append(question)
            return list(WEB)
        question = "What is the current price of gold?"
        async def handler(request):
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                provider = paid_provider(client)

                def planned(prompt):   # search + answer call + verifier call, as estimated before sending
                    answers, checking = spending_plan([provider], provider, prompt, Limits())
                    return 0.008 + sum(answers.values()) + checking

                smallest, largest = planned(make_prompt(question, [])), planned(" " * MAX_PROMPT_BYTES)
                self.assertLess(smallest, largest * 0.99)
                for budget, expected in ((largest * 0.99, []), (largest * 1.01, [question])):
                    searched.clear()
                    await run_v1(question, [provider], provider, CORPUS, Limits(max_estimated_usd=budget),
                                 search_fn=tracking, search_usd=0.008,
                                 claims_fn=extract_cited_claims, verify_fn=verify_semantic)
                    with self.subTest(budget=budget):
                        self.assertEqual(searched, expected)

    async def test_live_search_needs_a_price(self):
        # With the real search and no price entered, nothing is searched or asked.
        run = await run_v1("What is the current price of gold?", [FakeModel("m1", GOOD_1)], FakeVerifier(), CORPUS,
                           LIMITS, claims_fn=extract_cited_claims, verify_fn=verify_semantic, mode="fixture")
        self.assertEqual(run.outcome, "abstained")
        self.assertIn("search_usd_per_call", run.final_text)
        self.assertEqual(run.candidates, [])

    async def test_exact_judge_tie_is_kept_and_named(self):
        # Review case: p = 0.5 in both orders used to be reported as a preference for the first answer.
        run = await self.ask(judge_fn=judge_even)
        self.assertEqual(run.outcome, "verified_answer")
        self.assertEqual(run.tied_ids, ["m1", "m2"])
        self.assertIn("exactly even", run.judge.reason)
        self.assertIn("rule-based tie-break chose", run.judge.reason)
        self.assertNotIn("preferred m", run.judge.reason)
        self.assertIn("a fixed rule, not the judge, chose", run.uncertainty)

    async def test_tie_then_preference_in_a_three_answer_knockout(self):
        async def judge(question, a, b):
            p = 0.5 if len(calls) == 0 else 0.8
            calls.append((a, b))
            return {"p_a": p, "p_a_original_order": p, "p_a_swapped_order": p, "seconds": 0.1}
        calls = []
        third = "Water boils at 100 degrees Celsius at sea level, the standard reference pressure. [web_1]"
        models = [FakeModel("m1", GOOD_1), FakeModel("m2", GOOD_2), FakeModel("m3", third)]
        run = await self.ask(models=models, judge_fn=judge)
        self.assertEqual(len(run.judge.comparisons), 2)
        self.assertEqual(run.tied_ids, sorted(run.tied_ids))
        self.assertIn(run.selected_id, run.tied_ids)
        self.assertIn("then preferred", run.judge.reason)

    async def test_ties_are_named_round_by_round(self):
        # Four answers; the judge reads 0.5, then 0.8, then 0.5. The shortest answer wins each tie-break.
        readings = iter([0.5, 0.8, 0.5])
        async def judge(question, a, b):
            p = next(readings)
            return {"p_a": p, "p_a_original_order": p, "p_a_swapped_order": p, "seconds": 0.1}
        texts = ["Water boils at 100 degrees Celsius at sea level. [web_1]",
                 "Water boils at 100 degrees Celsius at sea level, as measured. [web_1]",
                 "Water boils at 100 degrees Celsius at sea level, as tables show. [web_1]",
                 "Water boils at 100 degrees Celsius at sea level, as recorded. [web_1]"]
        run = await self.ask(models=[FakeModel(f"m{i + 1}", t) for i, t in enumerate(texts)], judge_fn=judge)
        self.assertEqual(run.selected_id, "m1")
        self.assertEqual(run.tied_ids, ["m1", "m2", "m4"])
        self.assertEqual(run.judge.reason,
                         "Trained judge (both answer orders averaged) rated m1 and m2 exactly even (p = 0.50), "
                         "so the rule-based tie-break chose m1; then preferred m1 over m3 (p = 0.80); "
                         "then rated m1 and m4 exactly even (p = 0.50), so the rule-based tie-break chose m1")

    async def test_rule_fallback_ties_are_kept(self):
        texts = ["Water boils at 100 degrees Celsius at sea level. [web_1]",
                 "Water boils at 100 degrees Celsius at sea level, as measured. [web_1]"]
        run = await self.ask(models=[FakeModel(f"m{i + 1}", t) for i, t in enumerate(texts)], judge_fn=judge_down)
        self.assertEqual((run.selected_id, run.tied_ids), ("m1", ["m1", "m2"]))
        self.assertIn("fallback rule scored several answers equally", run.uncertainty)

    async def test_malformed_judge_reply_falls_back_to_rules(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as client:
            run = await self.ask(judge_fn=lambda q, a, b: judge_pair(q, a, b, client=client))
        self.assertEqual(run.outcome, "verified_answer")
        self.assertFalse(run.judge.used)
        self.assertIn("rule-based tie-break", run.judge.reason)


class HttpErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_status_is_reported_without_body(self):
        async def handler(request):
            return httpx.Response(404, json={"error": {"message": "secret-ish body text"}})
        config = ProviderConfig(provider="anthropic", model="missing-model", input_usd_per_million=1,
                                output_usd_per_million=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}):
                from answerlab.pipeline_v1 import collect_cited
                provider = HTTPProvider(config, client)
                provider.id = "anthropic:missing-model"
                [result] = await collect_cited([provider], "prompt", Limits(max_estimated_usd=1))
        self.assertEqual(result.status, "failed")
        self.assertIn("HTTP 404: model name not found", result.error)
        self.assertNotIn("secret-ish", result.model_dump_json())


class HelperTests(unittest.TestCase):
    def test_fit_evidence_shortens_instead_of_failing(self):
        long = [Passage(id=f"web_{i}", title="t", location="https://example.org", text="word " * 1100)
                for i in range(1, 6)]
        kept = fit_evidence("Q?", long)
        size = len(json.dumps({"question": "Q?", "evidence": [{"id": p.id, "text": p.text} for p in kept]}).encode())
        self.assertLessEqual(size, 22_000)
        self.assertGreaterEqual(len(kept), 3)

    def test_recheck_rejects_edited_text_and_changed_sources(self):
        claims = extract_cited_claims(GOOD_1)
        from answerlab.records import Check
        checks = [Check(claim=c, status="pass", reason="ok", source_hash=WEB[0].content_hash,
                        source_hashes={"web_1": WEB[0].content_hash}) for c in claims]
        self.assertTrue(recheck(GOOD_1, WEB, checks, extract_cited_claims))
        self.assertFalse(recheck(GOOD_1.replace("100", "90"), WEB, checks, extract_cited_claims))
        changed = [WEB[0].model_copy(update={"published_at": "2030-01-01"}), WEB[1]]
        self.assertFalse(recheck(GOOD_1, changed, checks, extract_cited_claims))

    def test_recheck_covers_every_cited_source(self):
        # Review case: a change to, or deletion of, the second cited source used to pass the recheck.
        import asyncio
        text = "Water boils at 100 degrees Celsius at sea level and at lower temperatures higher up. [web_1][web_2]"
        checks, _ = asyncio.run(verify_semantic(extract_cited_claims(text), WEB, FakeVerifier(), LIMITS))
        self.assertEqual(checks[0].status, "pass")
        self.assertTrue(recheck(text, WEB, checks, extract_cited_claims))
        changed = [WEB[0], WEB[1].model_copy(update={"text": "Altitude makes no difference."})]
        self.assertFalse(recheck(text, changed, checks, extract_cited_claims))
        self.assertFalse(recheck(text, WEB[:1], checks, extract_cited_claims))

    def test_strip_citations(self):
        self.assertEqual(strip_citations("A fact. [web_1][web_2] Another. [x]"), "A fact. Another.")

    def test_build_live_uses_models_with_keys_in_priority_order(self):
        config = LiveConfig(providers=[
            ProviderConfig(provider="anthropic", model="big", input_usd_per_million=1, output_usd_per_million=1),
            ProviderConfig(provider="openai", model="mid", input_usd_per_million=1, output_usd_per_million=1),
            ProviderConfig(provider="anthropic", model="small", input_usd_per_million=1, output_usd_per_million=1)],
            verifier=ProviderConfig(provider="openai", model="mini", input_usd_per_million=1, output_usd_per_million=1),
            limits=Limits(max_calls=2))
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"}, clear=True):
            providers, verifier, notes = build_live(config)
        self.assertEqual([p.id for p in providers], ["anthropic:big", "anthropic:small"])
        self.assertEqual(verifier.model, "big")   # equal prices: first usable model verifies
        self.assertTrue(any("OPENAI_API_KEY" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
