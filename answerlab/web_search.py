"""Step 5: retrieve source snippets with one bounded Tavily request.

API fields: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""

import asyncio
import json
import math
import os

import httpx

from .records import Passage


class SearchNotSent(ValueError):
    """The search was refused before any request was sent, so it cost nothing."""


async def web_search(query: str, limit: int = 5, timeout_seconds: float = 10.0,
                     client: httpx.AsyncClient | None = None) -> list[Passage]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise SearchNotSent("Set TAVILY_API_KEY in your environment before web search")
    if not query.strip():
        raise SearchNotSent("Search query must not be empty")
    if type(limit) is not int or limit < 0:
        raise SearchNotSent("Search limit must be a nonnegative integer")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise SearchNotSent("Search timeout must be positive and finite")
    if limit == 0:
        return []
    body = {"query": query, "max_results": min(limit, 20), "search_depth": "basic",
            "include_answer": False, "include_raw_content": False, "include_published_date": True}

    async def send(session):
        async with session.stream("POST", "https://api.tavily.com/search", json=body,
                                  headers={"Authorization": "Bearer " + key},
                                  timeout=timeout_seconds, follow_redirects=False) as response:
            response.raise_for_status()
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > 1_000_000:
                    raise ValueError("Search response exceeds 1 MB size limit")
            return json.loads(content)

    async with asyncio.timeout(timeout_seconds):
        if client is not None:
            data = await send(client)
        else:
            async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as session:
                data = await send(session)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Invalid search response: expected results list")
    passages = []
    for result in data["results"]:
        if not isinstance(result, dict):
            continue
        text = next((result[name].strip() for name in ("raw_content", "content", "snippet")
                     if isinstance(result.get(name), str) and result[name].strip()), "")
        url = result.get("url")
        if not text or not isinstance(url, str) or not url.strip() or len(url.strip()) > 2000:
            continue
        title = result.get("title")
        title = title.strip() if isinstance(title, str) else ""
        published = result.get("published_date")
        passages.append(Passage(id=f"web_{len(passages) + 1}", title=(title or url.strip())[:300],
                                text=text[:6000], location=url.strip(),
                                published_at=published if isinstance(published, str) else None,
                                fictional=False))
        if len(passages) >= min(limit, 20):
            break
    return passages
