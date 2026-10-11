"""Steps 1 and 2: small rules we can read before learning classifiers."""

import re

from .records import SourcePlan, Task


def understand(question: str) -> Task:
    question = question.strip()
    if not question or len(question) > 4000:
        raise ValueError("Enter a question between 1 and 4000 characters")
    lowered = question.casefold()
    kinds, reasons = [], []
    rules = (
        ("creative", r"\b(write|compose|invent)\b.*\b(story|poem|song|fiction)\b", "Creative-writing request"),
        ("document", r"\b(manual|document|passage|report|supplied|provided|attached)\b", "Refers to source material"),
        ("code", r"\b(code|debug|function|program|script)\b", "Programming terms detected"),
        ("calculation", r"\b(calculate|multiply|divide|sum)\b|\d\s*[+*/×]\s*\d", "Calculation requested"),
        ("comparison", r"\b(compare|comparison|versus|vs)\b", "Comparison requested"),
    )
    for kind, pattern, reason in rules:
        if re.search(pattern, lowered):
            kinds.append(kind)
            reasons.append(reason)
    if not kinds:
        kinds.append("general")
        reasons.append("No specialized rule matched; this does not prove the task is understood")
    ambiguous = len(re.findall(r"\w+", question)) < 3 or bool(
        re.fullmatch(r"(?:explain|summarize|fix|compare)\s+(?:this|it|these)[.!?]*", lowered)
    )
    return Task(
        question=question, kinds=kinds, reasons=reasons,
        needs_clarification=ambiguous,
        rubric=["Answer the requested question", "Preserve conditions and units", "Cite inspectable evidence", "Expose missing checks"],
    )


def decide_sources(task: Task) -> SourcePlan:
    text = task.question.casefold()
    if "creative" in task.kinds:
        return SourcePlan(mode="none", reason="Creative-only handling is outside this document prototype")
    # A supplied historical document is not replaced with today's web.
    if "document" in task.kinds and not re.search(r"\b(latest|newest|updated)\b", text):
        return SourcePlan(mode="saved_documents", reason="Use the identified document snapshot")
    if re.search(r"\b(today|current|currently|latest|now|weather|price|prices|news|ceo|president)\b", text):
        return SourcePlan(mode="fresh_web", reason="Potentially changing information requires fresh evidence")
    if "calculation" in task.kinds or "code" in task.kinds:
        return SourcePlan(mode="none", reason="This task needs specialized checks outside this first slice")
    return SourcePlan(mode="uncertain", reason="Try saved sources; make no freshness guarantee")
