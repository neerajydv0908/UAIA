import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from answerlab.providers import CITED_INSTRUCTIONS, INSTRUCTIONS, HTTPProvider, collect, normalize
from answerlab.records import Limits, ProviderConfig


class NormalizationTests(unittest.TestCase):
    def test_openai_text_blocks_and_usage(self):
        data = {"status": "completed", "model": "returned-version", "output": [
            {"content": [{"type": "output_text", "text": "one"}, {"type": "output_text", "text": "two"}]}],
            "usage": {"input_tokens": 12, "output_tokens": 3}}
        result = normalize("openai", "requested", data)
        self.assertEqual(result.text, "one\ntwo")
        self.assertEqual(result.model, "returned-version")
        self.assertEqual(result.input_tokens, 12)

    def test_missing_usage_not_zero(self):
        result = normalize("anthropic", "model", {"stop_reason": "end_turn", "content": [{"type": "text", "text": "hello"}]})
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.estimated_cost_usd)

    def test_refusal_and_truncation_remain_distinct(self):
        refusal = normalize("openai", "model", {"status": "completed", "output": [{"content": [{"type": "refusal"}]}]})
        truncated = normalize("anthropic", "model", {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "partial"}]})
        self.assertEqual(refusal.status, "refused")
        self.assertEqual(truncated.status, "truncated")

    def test_empty_success_is_failure(self):
        self.assertEqual(normalize("openai", "model", {"status": "completed", "output": []}).status, "failed")


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def config(self, provider="openai"):
        # Invented rates for testing arithmetic, not actual provider prices.
        return ProviderConfig(provider=provider, model="test-model", input_usd_per_million=1, output_usd_per_million=2)

    async def test_budget_blocks_before_http(self):
        calls = []
        async def handler(request):
            calls.append(request)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                with self.assertRaises(ValueError):
                    await collect([HTTPProvider(self.config(), client)], "question", Limits(max_estimated_usd=0))
        self.assertEqual(calls, [])

    async def test_rates_required_before_http(self):
        provider = HTTPProvider(ProviderConfig(provider="openai", model="test-model"))
        with self.assertRaises(ValueError):
            await collect([provider], "question", Limits(max_estimated_usd=1))

    async def test_two_requests_overlap_and_normalize(self):
        started = set()
        both = asyncio.Event()
        async def handler(request):
            started.add(request.url.host)
            if len(started) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), timeout=1)
            if request.url.host == "api.openai.com":
                return httpx.Response(200, json={"status": "completed", "output": [{"content": [{"type": "output_text", "text": "A"}]}]})
            return httpx.Response(200, json={"stop_reason": "end_turn", "content": [{"type": "text", "text": "B"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "ANTHROPIC_API_KEY": "test-only"}):
                results = await collect([HTTPProvider(self.config(p), client) for p in ("openai", "anthropic")],
                                        "question", Limits(max_estimated_usd=1))
        self.assertEqual([r.text for r in results], ["A", "B"])

    async def test_error_details_do_not_expose_credentials(self):
        async def handler(request):
            raise RuntimeError("sensitive-key-do-not-log")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                results = await collect([HTTPProvider(self.config(), client)], "question", Limits(max_estimated_usd=1))
        self.assertEqual(results[0].status, "failed")
        self.assertNotIn("sensitive-key", results[0].model_dump_json())


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    def config(self, provider="openai"):
        return ProviderConfig(provider=provider, model="test-model",
                              input_usd_per_million=1, output_usd_per_million=2)

    async def test_default_and_cited_instructions_on_both_providers(self):
        for provider_name in ("openai", "anthropic"):
            requests = []
            def handler(request):
                requests.append(json.loads(request.content))
                if provider_name == "openai":
                    return httpx.Response(200, json={"status": "completed", "output": [
                        {"content": [{"type": "output_text", "text": "answer"}]}]})
                return httpx.Response(200, json={"stop_reason": "end_turn",
                                                "content": [{"type": "text", "text": "answer"}]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "ANTHROPIC_API_KEY": "test-only"}):
                    provider = HTTPProvider(self.config(provider_name), client)
                    await provider.generate("query", Limits())
                    await provider.generate("query", Limits(), instructions=CITED_INSTRUCTIONS)
                    text, cost = await provider.complete("Verifier instruction", "query", Limits(max_estimated_usd=1))
            field = "instructions" if provider_name == "openai" else "system"
            self.assertEqual([r[field] for r in requests], [INSTRUCTIONS, CITED_INSTRUCTIONS, "Verifier instruction"])
            self.assertEqual(text, "answer")
            self.assertIsNone(cost)

    async def test_preflight_prices_actual_system_text_before_any_http(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                provider = HTTPProvider(self.config(), client)
                limits = Limits(max_estimated_usd=0.004)
                short = provider.preflight("query", limits, instructions="short")
                long = provider.preflight("query", limits, instructions="é" * 4000)
                self.assertAlmostEqual(long - short, (8000 - 5) / 1_000_000)
                with self.assertRaisesRegex(ValueError, "spending exceeds"):
                    await provider.complete("é" * 4000, "query", limits)
        self.assertEqual(calls, [])

    async def test_complete_cost_from_usage(self):
        def handler(request):
            self.assertEqual(request.extensions["timeout"]["read"], 2)
            return httpx.Response(200, json={"status": "completed", "output": [
                {"content": [{"type": "output_text", "text": "{}"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 3}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                text, cost = await HTTPProvider(self.config(), client).complete(
                    "JSON", "query", Limits(max_estimated_usd=1, call_timeout_seconds=2))
        self.assertEqual(text, "{}")
        self.assertAlmostEqual(cost, 0.000016)

    async def test_incomplete_refusal_and_oversized_responses_raise(self):
        responses = [
            httpx.Response(200, json={"status": "incomplete", "output": [
                {"content": [{"type": "output_text", "text": "partial"}]}]}),
            httpx.Response(200, json={"status": "completed", "output": [
                {"content": [{"type": "refusal"}]}]}),
            httpx.Response(200, content=b" " * 1_000_001),
            httpx.Response(500),
        ]
        for response in responses:
            with self.subTest(status=response.status_code):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
                    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                        with self.assertRaises((ValueError, httpx.HTTPStatusError)):
                            await HTTPProvider(self.config(), client).complete("verify", "query", Limits(max_estimated_usd=1))

    async def test_complete_enforces_total_timeout(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
                with self.assertRaises(TimeoutError):
                    await HTTPProvider(self.config(), client).complete(
                        "verify", "query", Limits(max_estimated_usd=1, request_timeout_seconds=0.01))


if __name__ == "__main__":
    unittest.main()
