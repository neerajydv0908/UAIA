# 6. Convert responses to a common format — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

- **Python dictionaries/dataclasses:** good for learning the record before adding a library.
- **Pydantic:** recommended candidate for typed validation and understandable errors. [Official guide](https://pydantic.dev/docs/validation/latest/get-started/).
- **Provider structured-output support:** useful where available; retain refusal/error handling and local validation. [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
- **LiteLLM:** optional common interface, with provider-specific checks still required. [Official documentation](https://docs.litellm.ai/docs/).
- **JSONL files initially; SQLite later:** one JSON record per line is convenient for small reproducible examples.

No LLM is needed to copy known provider fields into our schema. Reserve model calls for tasks requiring interpretation.

