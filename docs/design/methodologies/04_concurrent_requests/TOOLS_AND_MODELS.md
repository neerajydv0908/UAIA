# 4. Ask models concurrently — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Tool | Role | Recommendation |
| --- | --- | --- |
| Python asyncio | Schedule concurrent tasks and timeouts | First implementation candidate |
| Provider SDK async clients or HTTPX AsyncClient | Send network requests | Use one clear approach per adapter |
| LiteLLM | Common provider interface/gateway | Optional after direct integration is understood |
| LangGraph | Explicit workflow state and branching | Consider when the simple flow becomes hard to maintain |
| Database-backed worker jobs | Resume long requests and control duplicate work | Later product requirement if needed |

References: [asyncio tasks](https://docs.python.org/3/library/asyncio-task.html), [HTTPX](https://www.python-httpx.org/async/), [LiteLLM](https://docs.litellm.ai/docs/), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview).

Python TaskGroup cancels sibling tasks on certain unhandled failures. Our adapter boundary should turn expected provider failures into explicit outcomes when continued collection is intended.

MCP is a tool-connection protocol, not a replacement for this scheduler.

