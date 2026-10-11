"""Step 5: a tiny keyword retriever, usable directly or through MCP."""

import asyncio
import json
import re
import sys
from pathlib import Path

from .records import Passage

STOP_WORDS = set("a an the what is are does do of for to in and its according say says about from how with supplied manual document".split())


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP_WORDS


def load_corpus(path: Path) -> list[Passage]:
    if path.stat().st_size > 1_000_000:
        raise ValueError("Keep the first corpus under 1 MB")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) > 200:
        raise ValueError("The corpus must be a list of at most 200 passages")
    passages = [Passage.model_validate(row) for row in data]
    if len({p.id for p in passages}) != len(passages):
        raise ValueError("Passage IDs must be unique")
    return passages


def search(passages: list[Passage], question: str, limit: int = 3) -> list[Passage]:
    if not 1 <= limit <= 5:
        raise ValueError("Retrieval limit must be between 1 and 5")
    query = words(question)
    scored = [(len(query & words(p.text)), p) for p in passages]
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [p for score, p in scored if score > 0][:limit]


async def retrieve(path: Path, question: str, transport: str = "direct") -> list[Passage]:
    if transport == "direct":
        return search(load_corpus(path), question)
    if transport != "mcp":
        raise ValueError("Unknown retrieval transport")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "answerlab.mcp_server", "--corpus", str(path.resolve())],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    async with asyncio.timeout(10):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("search_documents", {"query": question, "limit": 3})
                if result.isError:
                    raise ValueError("MCP retrieval tool reported an error")
                content = "".join(block.text for block in result.content if block.type == "text")
                rows = json.loads(content)
                return [Passage.model_validate(row) for row in rows]
