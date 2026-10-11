"""The readable path through all 13 responsibilities.

Read the numbered comments in order. Step 5 deliberately runs before step 4
so every generator receives the same evidence.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .checks import extract_claims, final_check, judge, select, verify
from .providers import collect, make_prompt
from .records import Limits, Run, Trace, fingerprint
from .retrieval import retrieve
from .steps import decide_sources, understand


async def run_pipeline(question: str, providers: list, corpus: Path,
                       limits: Limits | None = None, transport: str = "direct",
                       mode: str = "fixture") -> Run:
    limits = limits or Limits()
    started = monotonic()
    trace = []

    def mark(step, name, detail, status="done"):
        trace.append(Trace(step=step, name=name, detail=detail, status=status))

    # 1. Understand the request with inspectable rules.
    task = understand(question)
    mark(1, "Understand", ", ".join(task.kinds) + "; rule-based, English-first")

    # 2. Decide which sources would be needed.
    source_plan = decide_sources(task)
    mark(2, "Information needs", source_plan.reason)

    # 3. Fixed provider set: a baseline before learning a router.
    selected = providers[:limits.max_calls]
    mark(3, "Route", f"Selected {len(selected)} configured providers; no learned router")

    blocking = None
    if task.needs_clarification:
        blocking = "Please specify the document and the question you want answered."
    elif source_plan.mode == "fresh_web":
        blocking = "Fresh web evidence is needed, but live web retrieval is not implemented yet."
    elif source_plan.mode == "none":
        blocking = "This first version handles questions about saved documents, not this task type."

    # 5 precedes 4: retrieve shared evidence before generating answers.
    evidence = []
    if not blocking:
        try:
            async with asyncio.timeout(limits.request_timeout_seconds):
                evidence = await retrieve(corpus, question, transport)
        except Exception as exc:
            blocking = f"Evidence retrieval failed ({type(exc).__name__}); no answer was verified."
    mark(5, "Retrieve evidence", f"{len(evidence)} passages via {transport}; saved source snapshots",
         "skipped" if blocking else "done")
    if not blocking and not evidence:
        blocking = "No matching evidence was found in the saved documents."

    # 4 and 6. Send concurrent calls and normalize their results.
    candidates = []
    remaining = limits.request_timeout_seconds - (monotonic() - started)
    if not blocking and remaining <= 0:
        blocking = "The request deadline was reached before generation."
    if not blocking:
        prompt = make_prompt(question, evidence)
        candidates = await collect(selected, prompt, limits.model_copy(update={"request_timeout_seconds": remaining}))
    mark(4, "Concurrent requests", f"{len(candidates)} provider outcomes; mode={mode}",
         "skipped" if blocking else "done")
    mark(6, "Normalize", "Original text, provider status, timing, and missing usage remain distinct",
         "skipped" if not candidates else "done")

    # 7. Extract every nonempty answer line; no dropped unchecked prose.
    claims = {c.id: extract_claims(c.text) for c in candidates}
    mark(7, "Extract claims", "One complete quoted passage per line; general prose extraction is pending", "limited")

    # 8. Check exact source quotations. Other forms of verification are unknown.
    checks = {c.id: verify(claims[c.id], evidence) for c in candidates}
    mark(8, "Verify", "Exact passage/citation matching only; authority, safety, math, code, and meaning are unchecked", "limited")

    # 9. Eligibility before ranking; no trained or prompted judge yet.
    judgments = [judge(c, checks[c.id]) for c in candidates]
    mark(9, "Judge", "Rules require a completed answer with every quotation checked; no truth probability", "limited")

    # 10. Preserve the chosen answer; synthesis is deliberately a later experiment.
    winner, tied_ids = select(candidates, judgments, question)
    mark(10, "Select", f"Selected {winner.id}" if winner else "No eligible candidate; abstain")

    # 11. Recheck this exact outgoing text and its evidence fingerprints.
    outgoing = winner.text if winner else ""
    verified = bool(winner and final_check(outgoing, evidence, checks[winner.id]))
    mark(11, "Reverify", "Final quotations and source versions match" if verified else "No final answer passed",
         "done" if verified else "skipped")
    if not verified:
        outgoing = blocking or "I could not verify an answer with this prototype's quotation checks."
    uncertainty = (
        "Quotation fidelity checked against saved passages. Relevance and completeness use simple rules; "
        "source authority, freshness, broader factual truth, and content safety have not been established."
        if verified else "Insufficient checked evidence; no confidence percentage is available."
    )
    if any(p.fictional for p in evidence):
        uncertainty += " This corpus contains fictional teaching material."

    # 12. The renderer adds a fixed explanation, not another model-written answer.
    mark(12, "Present", "Answer or abstention with citations and explicit checking limits")

    # 13. Feedback is a later explicit CLI action, not automatic training.
    mark(13, "Feedback", "Optional local feedback command; human review required before any training", "waiting")
    return Run(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), mode=mode,
        question=task.question, task=task, source_plan=source_plan, evidence=evidence,
        candidates=candidates, checks=checks, judgments=judgments,
        selected_id=winner.id if verified else None, tied_ids=tied_ids,
        final_text=outgoing, final_hash=fingerprint(outgoing),
        outcome="source_quotes" if verified else "abstained", uncertainty=uncertainty, trace=trace,
        estimated_cost_usd=sum(c.estimated_cost_usd or 0 for c in candidates),
        cost_complete=all(c.estimated_cost_usd is not None for c in candidates),
    )


def render(run: Run, show_trace: bool = True) -> str:
    lines = [f"Mode: {run.mode} | Result: {run.outcome}", "", run.final_text, "", run.uncertainty]
    if run.selected_id:
        cited = {check.claim.source_id for check in run.checks[run.selected_id] if check.status == "pass"}
        lines += ["", "Sources:"]
        lines += [f"[{p.id}] {p.title} — {p.location}" for p in run.evidence if p.id in cited]
    if show_trace:
        lines += ["", "What happened:"]
        lines += [f"{item.step:02}. {item.name}: {item.detail} ({item.status})" for item in run.trace]
    lines += ["", f"Recorded cost estimate: ${run.estimated_cost_usd:.6f}" +
              ("; some call costs are unknown" if not run.cost_complete else "")]
    return "\n".join(lines)
