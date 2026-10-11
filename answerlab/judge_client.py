"""Step 9 client: ask the local trained-judge server which of two answers is better.

The server (judge_server/) runs on this computer only. Any failure becomes
JudgeUnavailable so the pipeline can fall back to its rule-based tie-break.
"""

import math
import os
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "http://127.0.0.1:8765"


class JudgeUnavailable(Exception):
    pass


def judge_url(url: str | None = None) -> str:
    url = (url or os.environ.get("JUDGE_URL") or DEFAULT_URL).rstrip("/")
    parts = urlsplit(url)
    # Questions and answers never leave this computer through the judge.
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise JudgeUnavailable("The judge URL must point to this computer (http://127.0.0.1:<port>)")
    return url


def _json_object(response) -> dict | None:
    """The reply as a JSON object (a dict), or None for anything else: a list, plain text or broken JSON."""
    try:
        data = response.json()
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _number(value) -> bool:
    """True for a finite int or float. True/False are excluded: Python counts them as 1 and 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:   # an integer too large to fit in a float
        return False


async def _request(method: str, url: str, timeout_seconds: float, client, **kwargs) -> dict:
    try:
        if client is not None:
            response = await client.request(method, url, timeout=timeout_seconds, **kwargs)
        else:
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as fresh:
                response = await fresh.request(method, url, timeout=timeout_seconds, **kwargs)
    except httpx.HTTPError as exc:
        raise JudgeUnavailable(f"Judge server not reachable ({type(exc).__name__}); start it with ./run_v1.sh") from None
    if response.status_code != 200:
        error = (_json_object(response) or {}).get("error", "")
        detail = str(error)[:200] if error else ""
        raise JudgeUnavailable(f"Judge returned HTTP {response.status_code}" + (f": {detail}" if detail else ""))
    if len(response.content) > 100_000:
        raise JudgeUnavailable("Judge reply too large")
    # Check the reply's shape before reading any field from it.
    data = _json_object(response)
    if data is None:
        raise JudgeUnavailable("Judge reply is not a JSON object")
    return data


async def judge_pair(question: str, answer_a: str, answer_b: str, url: str | None = None,
                     timeout_seconds: float = 60.0, client: httpx.AsyncClient | None = None) -> dict:
    """Return {"p_a", "p_a_original_order", "p_a_swapped_order", "seconds", ...}; p_a = P(A better)."""
    data = await _request("POST", judge_url(url) + "/judge", timeout_seconds, client,
                          json={"question": question, "answer_a": answer_a, "answer_b": answer_b})
    for key in ("p_a", "p_a_original_order", "p_a_swapped_order"):
        value = data.get(key)
        if not _number(value) or not 0 <= value <= 1:
            raise JudgeUnavailable("Judge reply is missing a valid probability")
    if not _number(data.get("seconds", 0)) or data.get("seconds", 0) < 0:
        raise JudgeUnavailable("Judge reply has an invalid time")
    return data


async def judge_health(url: str | None = None, timeout_seconds: float = 3.0,
                       client: httpx.AsyncClient | None = None) -> dict | None:
    try:
        return await _request("GET", judge_url(url) + "/health", timeout_seconds, client)
    except JudgeUnavailable:
        return None
