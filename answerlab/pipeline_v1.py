"""The v1 live path through all 13 steps.

Evidence (web search or saved documents) -> answers that cite every sentence
-> each sentence checked against its cited passage -> the trained judge picks
between answers that passed -> recheck -> respond. Step 5 runs before step 4
so every answer model sees the same evidence.

pipeline.py keeps the original quotation-only demo unchanged.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .checks import select as rule_select
from .providers import CITED_INSTRUCTIONS, FixtureProvider, HTTPProvider, make_prompt
from .records import (Candidate, Check, JudgeComparison, JudgeRecord, Judgment, LiveConfig, Limits,
                      Passage, Run, Trace, cited_hashes, fingerprint)
from .retrieval import load_corpus, words
from .steps import decide_sources, understand
from .web_search import SearchNotSent

ROOT = Path(__file__).resolve().parent.parent
KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
MIN_PASS_SHARE = 0.8        # an answer may keep a few unconfirmed sentences, never a contradicted one
MAX_PROMPT_BYTES = 22_000   # make_prompt refuses anything above 24,000 bytes
LOCAL_MIN_OVERLAP = 2       # shared content words before a saved passage counts as clearly relevant
WEB_RESULTS = 5
CITATION_TAG = re.compile(r"\s*\[[A-Za-z0-9_-]{1,60}\]")
OVER_BUDGET = "Estimated spending for this question exceeds the configured budget"


class BudgetExceeded(ValueError):
    """A paid request was refused before sending, because the budget cannot cover it."""


def load_env_file(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines into the environment without printing or overriding anything."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def build_live(config: LiveConfig):
    """Answer models in priority order (first max_calls with a key) plus the verifier model."""
    notes = []
    usable = [p for p in config.providers if os.environ.get(KEY_NAMES[p.provider])]
    missing = sorted({KEY_NAMES[p.provider] for p in config.providers} - {KEY_NAMES[p.provider] for p in usable})
    if missing:
        notes.append("missing " + ", ".join(missing))
    providers = []
    for settings in usable[: config.limits.max_calls]:
        provider = HTTPProvider(settings)
        provider.id = f"{settings.provider}:{settings.model}"
        providers.append(provider)
    verifier_settings = config.verifier
    if verifier_settings is None or not os.environ.get(KEY_NAMES[verifier_settings.provider]):
        # Fall back to the cheapest model that has a key.
        verifier_settings = min(usable, key=lambda p: p.output_usd_per_million or float("inf")) if usable else None
        if config.verifier is not None and verifier_settings is not None:
            notes.append(f"verifier key missing; using {verifier_settings.model} to verify")
    verifier = HTTPProvider(verifier_settings) if verifier_settings else None
    return providers, verifier, notes


def prompt_bytes(question: str, passages: list[Passage]) -> int:
    return len(json.dumps({"question": question, "evidence": [{"id": p.id, "text": p.text} for p in passages]}).encode())


def fit_evidence(question: str, passages: list[Passage], max_bytes: int = MAX_PROMPT_BYTES) -> list[Passage]:
    """Keep passages in rank order until the prompt is full; shorten the last one rather than fail."""
    kept = []
    for passage in passages:
        if prompt_bytes(question, kept + [passage]) <= max_bytes:
            kept.append(passage)
            continue
        size = len(passage.text)
        while size > 200:
            size = int(size * 0.85)
            shorter = passage.model_copy(update={"text": passage.text[:size].rstrip() + " …"})
            if prompt_bytes(question, kept + [shorter]) <= max_bytes:
                kept.append(shorter)
                break
        break
    return kept


def strong_local_matches(corpus: Path, question: str, limit: int = 3) -> list[Passage]:
    query = words(question)
    scored = [(len(query & words(p.title + " " + p.text)), p) for p in load_corpus(corpus)]
    scored = [(score, p) for score, p in scored if score >= LOCAL_MIN_OVERLAP]
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [p for _, p in scored[:limit]]


async def gather_evidence(question: str, mode: str, corpus: Path, search_fn) -> tuple[list[Passage], str]:
    """Step 5 routing: saved documents, fresh web, or both when it is unclear."""
    local = strong_local_matches(corpus, question) if mode in ("saved_documents", "uncertain") else []
    if mode == "saved_documents":
        return local, f"{len(local)} saved passages"
    try:
        web = await search_fn(question)
    except Exception as exc:
        if mode == "fresh_web" or not local:
            raise
        if isinstance(exc, BudgetExceeded):
            why = "skipped (over budget)"
        elif isinstance(exc, SearchNotSent):
            why = f"skipped ({exc})"          # our own message, safe to show
        else:
            why = f"failed ({type(exc).__name__})"
        return local, f"{len(local)} saved passages; web search {why}"
    seen = {p.id for p in local}
    combined = local + [p for p in web if p.id not in seen]
    return combined, f"{len(local)} saved + {len(web)} web passages"


HTTP_HINTS = {400: "request rejected", 401: "check this API key", 403: "key lacks access",
              404: "model name not found", 429: "rate limit or no API credit", 529: "provider overloaded"}


def http_detail(exc: Exception) -> str:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(code, int):
        return type(exc).__name__
    hint = HTTP_HINTS.get(code, "provider error" if code >= 500 else "")
    return f"HTTP {code}" + (f": {hint}" if hint else "")


def spending_plan(providers: list, verifier, prompt: str, limits: Limits) -> tuple[dict[str, float], float]:
    """Conservative estimates before sending: each answer call, plus one verifier call per answer."""
    answers = {p.id: p.preflight(prompt, limits, CITED_INSTRUCTIONS) for p in providers if isinstance(p, HTTPProvider)}
    checking = 0.0
    if isinstance(verifier, HTTPProvider):
        # Sized for the evidence, a long answer and generous instructions.
        checking = len(providers) * verifier.preflight(prompt + " " * (4 * limits.max_output_tokens),
                                                       limits, " " * 2000)
    return answers, checking


async def collect_cited(providers: list, prompt: str, limits: Limits, reserve_usd: float = 0.0) -> list[Candidate]:
    """Step 4: concurrent cited-answer calls with one spending check before any request."""
    reservations = [p.preflight(prompt, limits, CITED_INSTRUCTIONS) for p in providers if isinstance(p, HTTPProvider)]
    if reservations and (limits.max_estimated_usd <= 0 or sum(reservations) + reserve_usd > limits.max_estimated_usd):
        raise BudgetExceeded(OVER_BUDGET + "; no answer-model calls sent")

    async def one(provider):
        started = monotonic()
        is_fixture = isinstance(provider, FixtureProvider)
        try:
            async with asyncio.timeout(min(limits.call_timeout_seconds, limits.request_timeout_seconds)):
                if is_fixture:
                    result = await provider.generate(prompt, limits)
                else:
                    result = await provider.generate(prompt, limits, instructions=CITED_INSTRUCTIONS)
        except TimeoutError:
            result = Candidate(id=provider.id, provider=provider.provider, model=provider.model, status="timeout",
                               error="Provider deadline exceeded", estimated_cost_usd=0 if is_fixture else None)
        except Exception as exc:
            # Exception bodies may contain keys, source text, or server responses: keep only the HTTP status.
            result = Candidate(id=provider.id, provider=provider.provider, model=provider.model, status="failed",
                               error=f"Provider request failed ({http_detail(exc)})",
                               estimated_cost_usd=0 if is_fixture else None)
        return result.model_copy(update={"id": provider.id, "elapsed_ms": round((monotonic() - started) * 1000, 2)})

    return list(await asyncio.gather(*(one(p) for p in providers)))


def eligibility(candidate: Candidate, checks: list[Check]) -> Judgment:
    total = len(checks)
    passed = sum(c.status == "pass" for c in checks)
    failed = sum(c.status == "fail" for c in checks)
    if candidate.status != "completed" or not total:
        return Judgment(candidate_id=candidate.id, eligible=False, passed=passed, total=total,
                        reason=f"Not eligible: answer {candidate.status}" if candidate.status != "completed"
                        else "Not eligible: no checkable sentences")
    if failed:
        return Judgment(candidate_id=candidate.id, eligible=False, passed=passed, total=total,
                        reason=f"Not eligible: {failed} sentence(s) contradict the cited source")
    if passed / total < MIN_PASS_SHARE:
        return Judgment(candidate_id=candidate.id, eligible=False, passed=passed, total=total,
                        reason=f"Not eligible: only {passed} of {total} sentences confirmed by their sources")
    return Judgment(candidate_id=candidate.id, eligible=True, passed=passed, total=total,
                    reason=(f"All {total} sentences supported" if total > 1 else "Its one sentence is supported")
                    if passed == total
                    else f"{passed} of {total} sentences supported; none contradicted")


def strip_citations(text: str) -> str:
    """The judge was trained on plain answers; evidence checking already happened in step 8."""
    return CITATION_TAG.sub("", text).strip()


async def choose(question: str, pool: list[Candidate], judgments: list[Judgment], judge_fn, time_left):
    """Steps 9-10: knockout between eligible answers using the trained judge (both orders).

    Returns (winner, judge record, tied IDs). A reading of exactly 0.5 is a tie, not a preference:
    the rule-based tie-break then picks one of the two, and the record says so.
    """
    if not pool:
        return None, JudgeRecord(used=False, reason="No eligible answer to compare"), []
    if len(pool) == 1:
        return pool[0], JudgeRecord(used=False, winner_id=pool[0].id,
                                    reason="Only one answer passed the checks, so the judge was not needed"), []
    from .judge_client import JudgeUnavailable
    pool_judgments = [j for j in judgments if j.candidate_id in {c.id for c in pool}]
    comparisons, rounds, champion, tied = [], [], pool[0], []
    try:
        if judge_fn is None:
            raise JudgeUnavailable("Judge disabled for this run")
        for challenger in pool[1:]:
            seconds = time_left()
            if seconds < 5:
                raise JudgeUnavailable("Not enough time left in this request for the judge")
            reading = await asyncio.wait_for(
                judge_fn(question, strip_citations(champion.text), strip_citations(challenger.text)),
                timeout=min(60.0, seconds))
            comparisons.append(JudgeComparison(
                a_id=champion.id, b_id=challenger.id, p_a=reading["p_a"],
                p_a_original_order=reading["p_a_original_order"], p_a_swapped_order=reading["p_a_swapped_order"],
                seconds=reading.get("seconds", 0)))
            if reading["p_a"] == 0.5:
                # Exactly even: the judge prefers neither answer. Keep the tie; a visible rule picks one.
                previous = champion
                champion, _ = rule_select([previous, challenger], pool_judgments, question)
                # tied = the current choice plus every answer the judge rated exactly even with it.
                tied = sorted({*tied, previous.id, challenger.id} if champion.id == previous.id
                              else {previous.id, challenger.id})
                rounds.append(f"rated {previous.id} and {challenger.id} exactly even (p = 0.50), "
                              f"so the rule-based tie-break chose {champion.id}")
            else:
                winner, loser = (champion, challenger) if reading["p_a"] > 0.5 else (challenger, champion)
                p = max(reading["p_a"], 1 - reading["p_a"])
                rounds.append(f"preferred {winner.id} over {loser.id} (p = {p:.2f})")
                if winner is challenger:
                    champion, tied = challenger, []
        last = comparisons[-1]
        p_winner = last.p_a if champion.id == last.a_id else 1 - last.p_a
        if not tied:
            reason = f"Trained judge preferred {champion.id} (p = {p_winner:.2f}, both answer orders averaged)"
        else:
            reason = "Trained judge (both answer orders averaged) " + "; then ".join(rounds)
        return champion, JudgeRecord(used=True, winner_id=champion.id, comparisons=comparisons,
                                     reason=reason), tied
    except (JudgeUnavailable, TimeoutError, KeyError, TypeError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, JudgeUnavailable) else (
            "Judge timed out" if isinstance(exc, TimeoutError) else f"Judge reply unusable ({type(exc).__name__})")
        winner, rule_ties = rule_select(pool, pool_judgments, question)
        return winner, JudgeRecord(used=False, winner_id=winner.id, comparisons=comparisons,
                                   reason=f"{reason}; used the rule-based tie-break instead"), rule_ties


def recheck(text: str, evidence: list[Passage], previous: list[Check], claims_fn) -> bool:
    """Step 11: the outgoing text must be exactly what was checked, against unchanged sources."""
    claims = claims_fn(text)
    if not claims or len(claims) != len(previous):
        return False
    passages = {p.id: p for p in evidence}
    for claim, check in zip(claims, previous):
        if (claim.text, claim.source_id) != (check.claim.text, check.claim.source_id) or check.status == "fail":
            return False
        # Every cited passage, not only the first, must still be there with the same content.
        if cited_hashes(claim.source_id, passages) != check.source_hashes:
            return False
    return True


async def run_v1(question: str, providers: list, verifier, corpus: Path, limits: Limits, *,
                 search_fn=None, judge_fn=None, claims_fn=None, verify_fn=None,
                 mode: str = "live", setup_notes: list[str] | None = None, search_usd: float | None = None) -> Run:
    if search_fn is None:
        from .web_search import web_search
        search_fn = lambda q: web_search(q, limit=WEB_RESULTS, timeout_seconds=10.0)  # noqa: E731
        search_price = search_usd                                 # None: no price entered for live search
    else:
        search_price = 0.0 if search_usd is None else search_usd  # a stand-in search is free unless priced
    if claims_fn is None:
        from .claims_cited import extract_cited_claims as claims_fn
    if verify_fn is None:
        from .verify_semantic import verify_semantic as verify_fn
    started = monotonic()
    trace = []

    def mark(step, name, detail, status="done"):
        trace.append(Trace(step=step, name=name, detail=detail, status=status))

    def time_left():
        return limits.request_timeout_seconds - (monotonic() - started)

    # spent: searches that succeeded; held: every search that may have been billed (counted against
    # the budget); unknown: a search failed after it may have been sent, so its cost is not known.
    search = {"spent": 0.0, "held": 0.0, "unknown": False}

    async def budgeted_search(query):
        """The web search costs money too. Send it only if the budget also covers the most expensive
        answer and verifier calls that can follow it, and count its cost."""
        if search_price is None:
            raise SearchNotSent("Enter the current web search price (search_usd_per_call) in the configuration "
                                "before live web search")
        # fit_evidence keeps every answer prompt within MAX_PROMPT_BYTES, so price the later calls at that size.
        largest = " " * max(MAX_PROMPT_BYTES, len(make_prompt(query, []).encode("utf-8")))
        answers, checking = spending_plan(providers, verifier, largest, limits)
        planned = search_price + sum(answers.values()) + checking
        if planned > 0 and (limits.max_estimated_usd <= 0 or planned > limits.max_estimated_usd):
            raise BudgetExceeded(OVER_BUDGET + "; nothing was sent")
        search["held"] += search_price
        try:
            found = await search_fn(query)
        except SearchNotSent:
            search["held"] -= search_price        # refused before sending: nothing to pay
            raise
        except BaseException:                     # includes being cancelled by the overall deadline
            search["unknown"] = search_price > 0  # it may or may not have been billed
            raise
        search["spent"] += search_price
        return found

    # 1. Understand the request with inspectable rules.
    task = understand(question)
    mark(1, "Understand", ", ".join(task.kinds) + "; rule-based, English-first")

    # 2. Decide whether the answer needs fresh information.
    plan = decide_sources(task)
    mark(2, "Freshness", plan.reason)

    # 3. Route: which evidence sources and which models.
    blocking = None
    if task.needs_clarification:
        blocking = "Please add a little more detail: what exactly should be answered?"
    elif plan.mode == "none":
        blocking = "This version answers factual questions from saved documents or the web. " + plan.reason + "."
    elif not providers:
        blocking = "No answer model is available. Add OPENAI_API_KEY and/or ANTHROPIC_API_KEY to .env."
    elif verifier is None:
        blocking = "No verifier model is available. Add an API key to .env."
    route = {"saved_documents": "saved documents", "fresh_web": "web search",
             "uncertain": "matching saved documents + web search"}.get(plan.mode, "none")
    detail = f"Evidence: {route}; answer models: {', '.join(p.model for p in providers) or 'none'}"
    detail += f"; verifier: {verifier.model if verifier else 'none'}"
    if setup_notes:
        detail += " (" + "; ".join(setup_notes) + ")"
    mark(3, "Route", detail, "skipped" if blocking else "done")

    # 5 precedes 4: gather shared evidence before generating answers.
    evidence, found = [], "not run"
    if not blocking:
        try:
            evidence, found = await asyncio.wait_for(gather_evidence(question, plan.mode, corpus, budgeted_search),
                                                     timeout=max(1.0, min(25.0, time_left())))
            evidence = fit_evidence(question, evidence)
        except BudgetExceeded as exc:
            blocking = str(exc)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            blocking = f"Evidence retrieval failed ({reason}); nothing was answered."
        if not blocking and not evidence:
            blocking = "No relevant evidence was found, so I won't guess."
    retrieved = f"{len(evidence)} passages used ({found})"
    if search["spent"]:
        retrieved += f"; web search about ${search['spent']:.4f}"
    elif search["unknown"]:
        retrieved += "; the failed web search may still have been billed"
    mark(5, "Retrieve evidence", retrieved, "skipped" if not evidence else "done")

    # 4 and 6. Concurrent cited-answer calls, normalized into candidate records.
    candidates, reserved = [], {}
    if not blocking:
        try:
            prompt = make_prompt(question, evidence)
            # Hold back budget for step 8 (one verifier call per answer) and count the search already sent.
            reserved, checking = spending_plan(providers, verifier, prompt, limits)
            call_limits = limits.model_copy(update={"request_timeout_seconds": max(1.0, time_left())})
            candidates = await collect_cited(providers, prompt, call_limits, checking + search["held"])
        except ValueError as exc:
            blocking = str(exc)
    mark(4, "Ask models in parallel", ", ".join(f"{c.id}: {c.status}" for c in candidates) or "no calls sent",
         "done" if candidates else "skipped")
    mark(6, "Normalize", "Text, status, timing, token usage and cost kept per model", "done" if candidates else "skipped")

    # 7. Split each completed answer into cited sentences.
    completed = [c for c in candidates if c.status == "completed"]
    claims = {c.id: claims_fn(c.text) for c in completed}
    mark(7, "Claims", ", ".join(f"{cid}: {len(items)} sentence{'s' if len(items) != 1 else ''}"
                                for cid, items in claims.items()) or "none",
         "done" if completed else "skipped")

    # 8. Check every sentence against the passage(s) it cites.
    checks = {c.id: [] for c in candidates}
    verify_cost, verify_cost_complete = 0.0, True
    if completed:
        # Unknown actual costs count at their (conservative) reserved estimate.
        spent = search["held"] + sum(c.estimated_cost_usd if c.estimated_cost_usd is not None
                                          else reserved.get(c.id, 0.0) for c in candidates)
        share = max(0.0, limits.max_estimated_usd - spent) / len(completed)
        verify_limits = limits.model_copy(update={"max_estimated_usd": share,
                                                  "request_timeout_seconds": max(1.0, time_left())})
        results = await asyncio.gather(*(verify_fn(claims[c.id], evidence, verifier, verify_limits) for c in completed),
                                       return_exceptions=True)
        for candidate, result in zip(completed, results):
            if isinstance(result, BaseException):
                verify_cost_complete = False
                checks[candidate.id] = [Check(claim=claim, status="unknown",
                                              reason=f"Verifier failed ({type(result).__name__})")
                                        for claim in claims[candidate.id]]
            else:
                checks[candidate.id], cost = result
                verify_cost += cost or 0.0
                verify_cost_complete &= cost is not None
    passed_total = {cid: (sum(c.status == "pass" for c in items), len(items)) for cid, items in checks.items() if items}
    mark(8, "Verify", ", ".join(f"{cid}: {p}/{t} supported" for cid, (p, t) in passed_total.items()) or "nothing to check",
         "done" if passed_total else "skipped")

    # 9. Eligibility, then the trained judge ranks the answers that passed.
    judgments = [eligibility(c, checks[c.id]) for c in candidates]
    eligible_ids = {j.candidate_id for j in judgments if j.eligible}
    eligible = [c for c in candidates if c.id in eligible_ids]
    fully = [c for c in eligible if all(check.status == "pass" for check in checks[c.id])]
    pool = fully or eligible
    winner, judge_record, tied = await choose(question, pool, judgments, judge_fn, time_left)
    mark(9, "Judge", judge_record.reason, "done" if judge_record.used else ("limited" if winner else "skipped"))

    # 10. Select the winner as written; merging answers is a later experiment.
    mark(10, "Select", f"Selected {winner.id}" if winner else "No answer passed the checks; abstain",
         "done" if winner else "skipped")

    # 11. Recheck the exact outgoing text and its source fingerprints.
    outgoing = winner.text if winner else ""
    rechecked = bool(winner) and recheck(outgoing, evidence, checks[winner.id], claims_fn)
    mark(11, "Recheck", "Outgoing sentences and source versions match the checked ones" if rechecked
         else "No answer passed the final recheck", "done" if rechecked else "skipped")

    if rechecked:
        passed = sum(c.status == "pass" for c in checks[winner.id])
        total = len(checks[winner.id])
        outcome = "verified_answer" if passed == total else "partly_verified"
        uncertainty = (f"Each sentence was checked against the passage it cites by {verifier.model} and judged supported. "
                       if passed == total else
                       f"{passed} of {total} sentences were confirmed by their cited sources; the others could not be "
                       "confirmed (none contradicted a source). ")
        uncertainty += "A model-based check is not proof, and the sources' own reliability was not assessed."
        if judge_record.used:
            uncertainty += " Judge probabilities are not calibrated."
        if judge_record.used and tied:
            uncertainty += " The judge rated the best answers exactly even; a fixed rule, not the judge, chose between them."
        elif tied:
            uncertainty += " Without the judge, the fallback rule scored several answers equally; the shortest was chosen."
    else:
        outcome = "abstained"
        outgoing = blocking or "No answer passed the evidence checks, so I won't guess."
        uncertainty = "No answer was verified; no confidence is claimed."
    if any(p.fictional for p in evidence):
        uncertainty += " Some evidence is fictional teaching material."

    # 12. Respond with the answer (or abstention), its sources, checks and limits.
    mark(12, "Respond", outcome.replace("_", " "))

    # 13. Feedback waits for an explicit, consented button press.
    mark(13, "Feedback", "Optional rating; stored only with consent and reviewed before any training", "waiting")

    generation_cost = sum(c.estimated_cost_usd or 0 for c in candidates)
    return Run(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), mode=mode,
        question=task.question, task=task, source_plan=plan, evidence=evidence,
        candidates=candidates, checks=checks, judgments=judgments,
        selected_id=winner.id if rechecked else None, tied_ids=tied if rechecked else [],
        final_text=outgoing, final_hash=fingerprint(outgoing), outcome=outcome,
        uncertainty=uncertainty, trace=trace,
        estimated_cost_usd=generation_cost + verify_cost + search["spent"],
        cost_complete=(all(c.estimated_cost_usd is not None for c in candidates) and verify_cost_complete
                       and not search["unknown"]),
        verifier=verifier.model if verifier else None, judge=judge_record,
    )


def render_v1(run: Run, show_trace: bool = True) -> str:
    lines = [f"Result: {run.outcome.replace('_', ' ')}", "", run.final_text, "", run.uncertainty]
    if run.selected_id:
        cited = {sid for check in run.checks[run.selected_id] for sid in (check.claim.source_id or "").split(",") if sid}
        lines += ["", "Sources:"] + [f"[{p.id}] {p.title} — {p.location}" for p in run.evidence if p.id in cited]
        lines += ["", "Sentence checks:"] + [f"  {c.status:7} {c.claim.text[:90]}" for c in run.checks[run.selected_id]]
    if show_trace:
        lines += ["", "What happened:"] + [f"{t.step:02}. {t.name}: {t.detail} ({t.status})" for t in run.trace]
    lines += ["", f"Estimated cost: ${run.estimated_cost_usd:.4f}" + ("" if run.cost_complete else " (some costs unknown)")]
    return "\n".join(lines)
