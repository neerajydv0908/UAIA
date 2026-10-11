import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from answerlab.claims_cited import extract_cited_claims
from answerlab.pipeline_v1 import run_v1
from answerlab.records import Limits
from answerlab.verify_semantic import verify_semantic
from tests.test_pipeline_v1 import CORPUS, GOOD_1, FakeModel, FakeVerifier, search
from webapp.server import App, make_server

ROOT = Path(__file__).resolve().parent.parent


async def fake_runner(question):
    return await run_v1(question, [FakeModel("m1", GOOD_1)], FakeVerifier(), CORPUS, Limits(max_estimated_usd=1),
                        search_fn=search, claims_fn=extract_cited_claims, verify_fn=verify_semantic, mode="fixture")


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.feedback = Path(cls.folder.name) / "feedback.jsonl"
        app = App(ROOT / "examples/live_config_v1.json", CORPUS, use_judge=False, runner=fake_runner,
                  feedback_path=cls.feedback)
        cls.server = make_server(app, port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.folder.cleanup()

    def request(self, method, path, body=None, host=None, content_type="application/json"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = content_type
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, payload

    def test_page_and_status(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"UAIA", body)
        status, body = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertIn("keys", json.loads(body))
        self.assertIn("search_price_usd", json.loads(body))

    def test_ask_then_feedback_with_consent_only(self):
        status, body = self.request("POST", "/api/ask", {"question": "How hot does water need to be to boil?"})
        self.assertEqual(status, 200)
        run = json.loads(body)
        self.assertEqual(run["outcome"], "verified_answer")
        status, _ = self.request("POST", "/api/feedback", {"run_id": run["id"], "label": "useful", "consent": False})
        self.assertEqual(status, 400)
        self.assertFalse(self.feedback.exists())
        status, body = self.request("POST", "/api/feedback", {"run_id": run["id"], "label": "useful", "consent": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["review_status"], "pending_human_review")
        self.assertNotIn("water", self.feedback.read_text())   # the question itself is not stored

    def test_unknown_run_feedback(self):
        status, _ = self.request("POST", "/api/feedback", {"run_id": "nope", "label": "useful", "consent": True})
        self.assertEqual(status, 404)

    def test_foreign_host_rejected(self):
        status, _ = self.request("GET", "/api/status", host="evil.example:8000")
        self.assertEqual(status, 403)

    def test_form_posts_rejected(self):
        status, _ = self.request("POST", "/api/ask", {"question": "x"}, content_type="text/plain")
        self.assertEqual(status, 415)


if __name__ == "__main__":
    unittest.main()
