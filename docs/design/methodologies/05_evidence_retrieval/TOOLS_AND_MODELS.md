# 5. Retrieve authoritative evidence — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Layer | Options | Proposed order |
| --- | --- | --- |
| Small local text search | SQLite FTS5 with keyword ranking | First retrieval baseline |
| Meaning-based retrieval | Sentence Transformers with BGE-M3; hosted OpenAI embeddings | After a measured keyword-retrieval gap |
| Reranking | BGE reranker from the [catalog](../MODEL_CATALOG.md) | After initial retrieval works |
| Larger persistent index | PostgreSQL with pgvector | When operational needs justify a database service |
| Web discovery | Perplexity Search API or Tavily | Trial one provider, then compare |
| Page text extraction | Trafilatura | Validate extracted passages against source pages |
| Standard tool interface | A configured read-only MCP search/retrieval server | After the direct tool behavior is clear |

Sources: [SQLite FTS5](https://www.sqlite.org/fts5.html), [retrieval and reranking](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html), [pgvector](https://github.com/pgvector/pgvector), [Perplexity Search](https://docs.perplexity.ai/docs/search/quickstart), [Tavily](https://docs.tavily.com/documentation/about), [Trafilatura](https://trafilatura.readthedocs.io/en/latest/), [MCP](https://modelcontextprotocol.io/docs/learn/architecture).

An MCP server exposes tools; it does not make a source authoritative. Use configured destinations and fetching limits; do not allow source text to request arbitrary tool actions.

