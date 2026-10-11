import hashlib
import json
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

import httpx

from answerlab.judge_client import JudgeUnavailable, judge_health, judge_pair
from judge_server.server import (RUBRIC, RUBRIC_SHA256, build_parser, compare_answers, is_out_of_memory,
                                 make_server, messages, pick_device, retry_plan)


def fake_tokenize(chat):
    # Encode which answer is shown first so the fake scorer can "prefer" one.
    answers = json.loads(chat[1]["content"])["answers"]
    return [1 if answers["A"] == "good" else 2] * 10


def fake_score(ids):
    return 0.8 if ids[0] == 1 else 0.3


class PromptTests(unittest.TestCase):
    def test_rubric_is_the_trained_rubric(self):
        self.assertEqual(hashlib.sha256(RUBRIC.encode()).hexdigest(), RUBRIC_SHA256)

    def test_messages_match_training_payload(self):
        chat = messages("Q?", "first", "second")
        self.assertEqual(chat[0], {"role": "system", "content": RUBRIC})
        self.assertEqual(chat[1]["content"],
                         json.dumps({"question": "Q?", "evidence": [], "answers": {"A": "first", "B": "second"}},
                                    ensure_ascii=False))

    def test_both_orders_are_averaged(self):
        result = compare_answers("Q?", "good", "bad", fake_tokenize, fake_score)
        self.assertAlmostEqual(result["p_a_original_order"], 0.8)
        self.assertAlmostEqual(result["p_a_swapped_order"], 0.7)
        self.assertAlmostEqual(result["p_a"], 0.75)

    def test_long_prompts_are_refused_not_truncated(self):
        with self.assertRaises(ValueError):
            compare_answers("Q?", "good", "bad", lambda chat: [1] * 5000, fake_score)

    def test_non_finite_score_is_an_error(self):
        with self.assertRaises(RuntimeError):
            compare_answers("Q?", "good", "bad", fake_tokenize, lambda ids: float("nan"))


class FakeJudge:
    def compare(self, question, a, b):
        return compare_answers(question, a, b, fake_tokenize, fake_score)

    def info(self):
        return {"status": "ok", "adapter": "fake"}


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(FakeJudge(), port=0)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, body, content_type="application/json"):
        request = urllib.request.Request(self.url + "/judge", data=json.dumps(body).encode(),
                                         headers={"Content-Type": content_type}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as response:
            self.assertEqual(json.loads(response.read())["status"], "ok")

    def test_judge_round_trip(self):
        status, data = self.post({"question": "Q?", "answer_a": "good", "answer_b": "bad"})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(data["p_a"], 0.75)

    def test_missing_field_rejected(self):
        status, data = self.post({"question": "Q?", "answer_a": "good"})
        self.assertEqual(status, 422)

    def test_non_json_content_type_rejected(self):
        status, _ = self.post({"question": "Q?", "answer_a": "a", "answer_b": "b"}, "text/plain")
        self.assertEqual(status, 415)

    def test_only_local_binding(self):
        with self.assertRaises(ValueError):
            make_server(FakeJudge(), host="0.0.0.0", port=0)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await judge_pair("Q?", "a", "b", client=client, **kwargs)

    async def test_valid_reply(self):
        reply = {"p_a": 0.6, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5, "seconds": 1.0}
        result = await self.call(lambda request: httpx.Response(200, json=reply))
        self.assertEqual(result["p_a"], 0.6)

    async def test_server_error_becomes_unavailable(self):
        with self.assertRaises(JudgeUnavailable):
            await self.call(lambda request: httpx.Response(500, json={"error": "Judge failed (RuntimeError)"}))

    async def test_invalid_probability_rejected(self):
        reply = {"p_a": 1.5, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5}
        with self.assertRaises(JudgeUnavailable):
            await self.call(lambda request: httpx.Response(200, json=reply))

    async def test_remote_judge_url_refused(self):
        with self.assertRaises(JudgeUnavailable):
            await self.call(lambda request: httpx.Response(200, json={}), url="https://example.com")

    async def test_unreachable_health_is_none(self):
        def handler(request):
            raise httpx.ConnectError("refused")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            self.assertIsNone(await judge_health(client=client))

    async def test_replies_that_are_not_json_objects_become_unavailable(self):
        # Review case: a JSON list such as [] used to crash with AttributeError instead of falling back.
        bodies = [b"[]", b"[1, 2]", b'"text"', b"null", b"not json", b"[" * 5000 + b"]" * 5000]
        for status in (200, 500):
            for body in bodies:
                with self.subTest(status=status, body=body[:12]):
                    with self.assertRaises(JudgeUnavailable):
                        await self.call(lambda request, s=status, b=body: httpx.Response(s, content=b))
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as client:
            self.assertIsNone(await judge_health(client=client))

    async def test_true_false_and_bad_times_are_not_numbers(self):
        replies = [{"p_a": True, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5},
                   {"p_a": 0.6, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5, "seconds": "slow"},
                   {"p_a": 0.6, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5, "seconds": -1},
                   # Integers too large for a float used to raise OverflowError and crash the run.
                   {"p_a": 10 ** 400, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5},
                   {"p_a": 0.6, "p_a_original_order": 0.7, "p_a_swapped_order": 0.5, "seconds": 10 ** 400}]
        for reply in replies:
            with self.subTest(reply=reply):
                with self.assertRaises(JudgeUnavailable):
                    await self.call(lambda request, r=reply: httpx.Response(200, json=r))


def fake_torch(cuda=False, mps=False):
    """Just enough of torch to answer "which devices does this computer have?"."""
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_nvidia_then_apple_then_cpu(self):
        # Review case: an NVIDIA (CUDA) machine used to fall back to the CPU.
        self.assertEqual(pick_device(fake_torch(cuda=True, mps=True)), "cuda")
        self.assertEqual(pick_device(fake_torch(cuda=True)), "cuda")
        self.assertEqual(pick_device(fake_torch(mps=True)), "mps")
        self.assertEqual(pick_device(fake_torch()), "cpu")

    def test_requested_device_is_used_or_refused(self):
        self.assertEqual(pick_device(fake_torch(cuda=True), "cpu"), "cpu")
        self.assertEqual(pick_device(fake_torch(cuda=True), "cuda"), "cuda")
        with self.assertRaises(ValueError):
            pick_device(fake_torch(mps=True), "cuda")
        with self.assertRaises(ValueError):
            pick_device(fake_torch(), "tpu")

    def test_small_nvidia_gpu_falls_back_to_the_cpu(self):
        oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertTrue(is_out_of_memory(fake_torch(cuda=True), oom))
        self.assertFalse(is_out_of_memory(fake_torch(cuda=True), RuntimeError("bfloat16 is not supported")))
        self.assertEqual(retry_plan("bfloat16", "cuda", "auto", True), ("bfloat16", "cpu"))
        self.assertEqual(retry_plan("bfloat16", "cuda", "cuda", False), ("float32", "cuda"))
        self.assertEqual(retry_plan("bfloat16", "mps", "auto", False), ("float32", "mps"))
        self.assertIsNone(retry_plan("float32", "cuda", "cuda", True))
        self.assertIsNone(retry_plan("bfloat16", "cuda", "cuda", True))   # float32 would need twice the memory
        self.assertIsNone(retry_plan("bfloat16", "mps", "auto", True))
        self.assertIsNone(retry_plan("float32", "mps", "auto", False))

    def test_start_command_accepts_a_device(self):
        self.assertEqual(build_parser().parse_args([]).device, "auto")
        self.assertEqual(build_parser().parse_args(["--device", "cuda"]).device, "cuda")


if __name__ == "__main__":
    unittest.main()
