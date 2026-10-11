"""A real, read-only MCP server with one configured local corpus."""

import argparse
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .retrieval import load_corpus, search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    passages = load_corpus(args.corpus)
    server = FastMCP("UAIA local evidence", log_level="WARNING")

    @server.tool()
    def search_documents(query: str, limit: int = 3) -> str:
        """Find passages in the configured corpus. This tool cannot change files."""
        if not query.strip() or len(query) > 4000:
            raise ValueError("Invalid query length")
        return json.dumps([p.model_dump() for p in search(passages, query, limit)])

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
