import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from answerlab.web_search import web_search


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_passage_mapping(self):
        calls = []
        async def handler(request):
            calls.append(request)
            self.assertEqual(str(request.url), "https://api.tavily.com/search")
            self.assertEqual(request.headers["Authorization"], "Bearer test-only")
            body = json.loads(request.content)
            self.assertEqual(body["query"], "weather update")
            self.assertEqual(body["max_results"], 2)
            self.assertTrue(body["include_published_date"])
            self.assertFalse(body["include_answer"])
            self.assertEqual(request.extensions["timeout"]["read"], 3)
            return httpx.Response(200, json={"results": [
                {"title": "empty", "url": "https://example.org/empty", "content": " "},
                {"title": "T" * 350, "url": "https://example.org/one", "content": "snippet",
                 "raw_content": "x" * 6500, "published_date": "2026-10-08"},
                {"title": "Two", "url": "https://example.org/two", "raw_content": None,
                 "content": "  A useful snippet.  "},
                {"title": "Three", "url": "https://example.org/three", "content": "unused"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
                result = await web_search("weather update", 2, 3, client)
            self.assertFalse(client.is_closed)
        self.assertEqual(len(calls), 1)
        self.assertEqual([p.id for p in result], ["web_1", "web_2"])
        self.assertEqual(len(result[0].title), 300)
        self.assertEqual(len(result[0].text), 6000)
        self.assertEqual(result[0].published_at, "2026-10-08")
        self.assertEqual(result[1].text, "A useful snippet.")
        self.assertIsNone(result[1].published_at)
        self.assertTrue(all(not p.fictional for p in result))

    async def test_missing_key_blocks_before_http(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "^Set TAVILY_API_KEY in your environment before web search$"):
                await web_search("query")

    async def test_empty_malformed_entries_and_api_limit(self):
        async def handler(request):
            self.assertEqual(json.loads(request.content)["max_results"], 20)
            return httpx.Response(200, json={"results": [None, {}, {"content": "no URL"},
                {"url": "https://example.org", "title": None, "snippet": "fallback"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
                result = await web_search("query", 30, client=client)
        self.assertEqual(result[0].title, "https://example.org")
        self.assertEqual(result[0].text, "fallback")

    async def test_invalid_arguments_and_zero_limit_do_not_call(self):
        def handler(request):
            self.fail("Invalid/zero-limit search must not make a request")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
                self.assertEqual(await web_search("query", 0, client=client), [])
                for args in [(" ", 5, 10), ("query", -1, 10), ("query", 2.5, 10),
                             ("query", 5, 0), ("query", 5, float("inf"))]:
                    with self.subTest(args=args), self.assertRaises(ValueError):
                        await web_search(*args, client=client)

    async def test_failures_are_not_empty_success(self):
        responses = [httpx.Response(503), httpx.Response(200, content=b"not JSON"),
                     httpx.Response(200, json={"results": {}}),
                     httpx.Response(200, content=b" " * 1_000_001)]
        for response in responses:
            with self.subTest(status=response.status_code):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
                    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
                        with self.assertRaises((ValueError, httpx.HTTPStatusError)):
                            await web_search("query", client=client)

    async def test_redirect_is_not_followed_even_with_supplied_client(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"location": "https://example.org/redirect"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
                with self.assertRaises(httpx.HTTPStatusError):
                    await web_search("query", client=client)
        self.assertEqual(len(calls), 1)

    async def test_total_deadline(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json={"results": []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
                with self.assertRaises(TimeoutError):
                    await web_search("query", timeout_seconds=0.01, client=client)

    async def test_owned_client_disables_environment_and_redirects(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": []})))
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-only"}):
            with patch("answerlab.web_search.httpx.AsyncClient", return_value=client) as factory:
                self.assertEqual(await web_search("query"), [])
                factory.assert_called_once_with(follow_redirects=False, trust_env=False)
        self.assertTrue(client.is_closed)


if __name__ == "__main__":
    unittest.main()
