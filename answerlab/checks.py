"""Steps 7–11: strict quotation checks and a transparent baseline judge.

This deliberately does NOT claim to verify paraphrases, arbitrary facts, code,
math, completeness, source authority, or safety. Those remain unimplemented.
"""

import re

from .records import Candidate, Check, Claim, Judgment, Passage, fingerprint
from .retrieval import words


def extract_claims(text: str) -> list[Claim]:
    claims = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"(.+?)\s+\[([A-Za-z0-9_-]{1,60})\]", line.strip())
        claims.append(Claim(text=match[1] if match else line.strip(),
                            source_id=match[2] if match else None, original_line=line))
    return claims


def verify(claims: list[Claim], evidence: list[Passage]) -> list[Check]:
    sources = {p.id: p for p in evidence}
    results = []
    for claim in claims:
        source = sources.get(claim.source_id)
        if source is None:
            result = Check(claim=claim, status="unknown", reason="Missing or unknown citation")
        elif claim.text != source.text:
            result = Check(claim=claim, status="unknown", source_hash=source.content_hash,
                           reason="Not an exact complete passage; meaning/factuality is unchecked")
        else:
            result = Check(claim=claim, status="pass", source_hash=source.content_hash,
                           reason="Exact complete passage matches the cited source snapshot")
        results.append(result)
    return results


def judge(candidate: Candidate, checks: list[Check]) -> Judgment:
    passed = sum(c.status == "pass" for c in checks)
    eligible = candidate.status == "completed" and bool(checks) and passed == len(checks)
    return Judgment(candidate_id=candidate.id, eligible=eligible, passed=passed, total=len(checks),
                    reason="Eligible source quotations; relevance/completeness remain heuristic" if eligible
                    else "Not eligible: incomplete response, missing claims, or unchecked content")


def select(candidates: list[Candidate], judgments: list[Judgment], question: str):
    eligible_ids = {j.candidate_id for j in judgments if j.eligible}
    eligible = [c for c in candidates if c.id in eligible_ids]
    if not eligible:
        return None, []
    # A ranking heuristic only. More overlap does not prove relevance or truth.
    scores = {c.id: len(words(question) & words(c.text)) for c in eligible}
    best = max(scores.values())
    ties = [c for c in eligible if scores[c.id] == best]
    # Stable, transparent tie-breaking without considering provider branding.
    winner = min(ties, key=lambda c: (len(c.text), fingerprint(c.text), c.id))
    return winner, [c.id for c in ties] if len(ties) > 1 else []


def final_check(text: str, evidence: list[Passage], previous: list[Check]) -> bool:
    checks = verify(extract_claims(text), evidence)
    if not checks or not all(c.status == "pass" for c in checks):
        return False
    # A rewritten claim or changed source cannot reuse an earlier pass.
    old = [(c.claim.text, c.claim.source_id, c.source_hash) for c in previous if c.status == "pass"]
    new = [(c.claim.text, c.claim.source_id, c.source_hash) for c in checks]
    return new == old
