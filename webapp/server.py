"""Steps 12-13: a local web page for the v1 pipeline. Listens on this computer only.

Run:  .venv/bin/python -m webapp      (./run_v1.sh starts the judge too)
"""

import argparse
import asyncio
import json
import os
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from answerlab.feedback import LABELS, record_feedback
from answerlab.judge_client import judge_health, judge_pair
from answerlab.pipeline_v1 import KEY_NAMES, ROOT, build_live, load_env_file, run_v1
from answerlab.records import LiveConfig, Run

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG = ROOT / "examples/live_config_v1.json"
FEEDBACK_PATH = ROOT / "data/private/feedback.jsonl"
MAX_BODY_BYTES = 20_000
KEEP_RUNS = 50   # answers stay in memory only; nothing is saved unless feedback is sent


class App:
    def __init__(self, config_path: Path, corpus: Path, use_judge: bool = True, runner=None,
                 feedback_path: Path = FEEDBACK_PATH):
        self.config = LiveConfig.model_validate_json(config_path.read_text())
        self.corpus = corpus
        self.use_judge = use_judge
        self.runner = runner or self.live_runner
        self.feedback_path = feedback_path
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.lock = threading.Lock()

    async def live_runner(self, question: str) -> Run:
        providers, verifier, notes = build_live(self.config)
        return await run_v1(question, providers, verifier, self.corpus, self.config.limits,
                            judge_fn=judge_pair if self.use_judge else None, setup_notes=notes,
                            search_usd=self.config.search_usd_per_call)

    def ask(self, question: str) -> Run:
        run = asyncio.run(self.runner(question))
        with self.lock:
            self.runs[run.id] = run
            while len(self.runs) > KEEP_RUNS:
                self.runs.popitem(last=False)
        return run

    def feedback(self, data: dict) -> dict:
        with self.lock:
            run = self.runs.get(str(data.get("run_id")))
        if run is None:
            raise LookupError("That answer is no longer in memory; ask again to rate it")
        note = data.get("note") or ""
        if not isinstance(note, str):
            raise ValueError("The note must be text")
        if data.get("consent") is not True:
            raise ValueError("Feedback is stored only when you agree to store it on this computer")
        return record_feedback(run, self.feedback_path, str(data.get("label")), note,
                               data.get("consent") is True, data.get("allow_training") is True)

    def status(self) -> dict:
        judge = asyncio.run(judge_health()) if self.use_judge else None
        keys = {name: bool(os.environ.get(name)) for name in sorted(set(KEY_NAMES.values()) | {"TAVILY_API_KEY"})}
        providers, verifier, notes = build_live(self.config)
        return {"judge": judge, "judge_enabled": self.use_judge, "keys": keys,
                "answer_models": [p.model for p in providers], "verifier": verifier.model if verifier else None,
                "notes": notes, "budget_usd": self.config.limits.max_estimated_usd,
                "search_price_usd": self.config.search_usd_per_call,
                "feedback_labels": sorted(LABELS)}


class Handler(BaseHTTPRequestHandler):
    server_version = "UAIA/1"

    def log_message(self, format, *args):   # questions and answers are not logged
        pass

    def send_body(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code: int, data):
        self.send_body(code, json.dumps(data).encode("utf-8"), "application/json")

    def local_host(self) -> bool:
        # Blocks DNS-rebinding pages that pretend to be this server.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")

    def do_GET(self):
        if not self.local_host():
            return self.send_json(403, {"error": "Local use only"})
        if self.path in ("/", "/index.html"):
            return self.send_body(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self.send_json(200, self.server.app.status())
        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self.local_host():
            return self.send_json(403, {"error": "Local use only"})
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self.send_json(415, {"error": "Send JSON"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            return self.send_json(413, {"error": "Request is empty or too large"})
        try:
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Send a JSON object")
            if self.path == "/api/ask":
                question = data.get("question")
                if not isinstance(question, str):
                    raise ValueError("Enter a question")
                return self.send_json(200, self.server.app.ask(question).model_dump(mode="json"))
            if self.path == "/api/feedback":
                return self.send_json(200, self.server.app.feedback(data))
            return self.send_json(404, {"error": "Not found"})
        except LookupError as exc:
            return self.send_json(404, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send_json(400, {"error": str(exc)[:300]})
        except Exception as exc:   # never echo internals, keys or source text
            return self.send_json(500, {"error": f"Something failed ({type(exc).__name__}); see the terminal"})


def make_server(app: App, port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = app
    return server


def main():
    parser = argparse.ArgumentParser(description="UAIA (U ask, I answer) web page, this computer only")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--corpus", type=Path, default=ROOT / "examples/corpus.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-judge", action="store_true", help="Use the rule-based tie-break instead of the judge")
    args = parser.parse_args()
    load_env_file()
    app = App(args.config, args.corpus, use_judge=not args.no_judge)
    server = make_server(app, args.port)
    status = app.status()
    print(f"Answer models: {', '.join(status['answer_models']) or 'none (add keys to .env)'}; "
          f"verifier: {status['verifier'] or 'none'}; web search key: {status['keys']['TAVILY_API_KEY']}; "
          f"judge: {'ready' if status['judge'] else 'not reachable'}", flush=True)
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
