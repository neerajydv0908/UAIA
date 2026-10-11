# 5. Retrieve authoritative evidence — methods

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Alternative strategies

| Method | Plain explanation | Strength | Weakness / cost |
| --- | --- | --- | --- |
| Curated evidence package | A human supplies the relevant passages | Best teaching and diagnostic baseline | Does not scale; human selection effort |
| Keyword retrieval / BM25 | Match informative words and rank passages | Good for exact names, codes, and terminology | Can miss paraphrases |
| Dense retrieval | Convert text to number lists called embeddings; find nearby meanings | Can match different wording | Similar text may contradict the answer |
| Hybrid retrieval + reranking | Combine keyword/meaning searches; reread a shortlist more carefully | Can combine complementary evidence | More components and runtime cost |
| Query rewriting / multi-hop retrieval | Reformulate or break the question into linked searches | Helps questions requiring several facts | May drift from the question or create too many calls |
| Graph retrieval | Retrieve connected entities and relationships | Potential fit for a structured domain | Building and maintaining the graph is substantial work |

BM25 is a keyword-ranking method. A reranker is a second model that reorders a short list of retrieved passages. A knowledge graph records entities and their relationships.

## Web strategy is a separate choice

A search API discovers pages; page retrieval/extraction obtains usable text. Provider-generated research answers can be candidates, but are not substitutes for inspected underlying evidence.

Prefer sources appropriate to the claim: original documentation for product behavior, primary research for a study, and dated official records for current published figures. Track conflicting sources rather than silently choosing the majority.

Deduplicate copies of the same origin. Search snippets are discovery aids; mark claims unverified if the necessary source text cannot be inspected. Treat embedded instructions as source text, never as application commands.

