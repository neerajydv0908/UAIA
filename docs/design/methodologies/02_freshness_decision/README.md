# 2. Decide whether fresh information is needed

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## What this step does

It decides which information the answer must use: supplied documents, a saved source collection, fresh web results, or no retrieval.

**Freshness** asks whether information could have changed. **Retrieval** means finding information. A historical document question needs retrieval even when it does not need today's web.

**Input:** original question, task record, relevant date/location, and available documents.

**Output:** source mode, reason, required “as of” date, and an uncertainty flag. Later retrieval must report whether that need was actually met.

## Simple example

“What is 12 × 8?” needs a calculation. “What is this service's current subscription price?” needs a current official source. “What did this 2020 contract say?” needs that contract version, not a replacement with today's terms.

If the user forbids web access, honor that constraint and explain any resulting gap.

## Connection to the papers

Activation policies motivate spending on useful additional work. PDE-Agents motivates supplying relevant evidence early. Neither paper gives us a ready-made general freshness classifier; these source-selection rules are our design.

