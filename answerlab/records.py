"""Shared records: the named boxes passed between our 13 steps."""

from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def fingerprint(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class Task(Record):
    question: str
    kinds: list[str]
    reasons: list[str]
    needs_clarification: bool = False
    rubric: list[str]


class SourcePlan(Record):
    mode: Literal["saved_documents", "fresh_web", "none", "uncertain"]
    reason: str


class Passage(Record):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,60}$")
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=6000)
    location: str = Field(min_length=1, max_length=2000)
    published_at: str | None = None
    fictional: bool = False

    @property
    def content_hash(self) -> str:
        return fingerprint(self.model_dump_json())


class Candidate(Record):
    id: str
    provider: str
    model: str
    text: str = ""
    status: Literal["completed", "failed", "timeout", "refused", "truncated", "skipped"]
    error: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_ms: float = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    fixture: bool = False


class Claim(Record):
    text: str
    source_id: str | None
    original_line: str


def cited_ids(source_id: str | None) -> list[str]:
    """The passage IDs a claim cites: "web_1,web_2" becomes ["web_1", "web_2"]."""
    return [part.strip() for part in source_id.split(",")] if source_id else []


def cited_hashes(source_id: str | None, passages: dict[str, Passage]) -> dict[str, str]:
    """The fingerprint of every cited passage that exists, keyed by its ID."""
    return {pid: passages[pid].content_hash for pid in cited_ids(source_id) if pid in passages}


class Check(Record):
    claim: Claim
    status: Literal["pass", "fail", "unknown"]
    reason: str
    source_hash: str | None = None          # the first cited passage (the quotation demo cites only one)
    # v1: every cited passage's fingerprint, so the final recheck notices a change in any of them.
    source_hashes: dict[str, str] = Field(default_factory=dict)


class Judgment(Record):
    candidate_id: str
    eligible: bool
    passed: int
    total: int
    reason: str


class Trace(Record):
    step: int
    name: str
    status: Literal["done", "limited", "skipped", "waiting"] = "done"
    detail: str


class JudgeComparison(Record):
    """One trained-judge reading. p_a is P(answer A is better), averaged over both answer orders."""
    a_id: str
    b_id: str
    p_a: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_a_original_order: float = Field(ge=0, le=1, allow_inf_nan=False)
    p_a_swapped_order: float = Field(ge=0, le=1, allow_inf_nan=False)
    seconds: float = Field(default=0, ge=0)


class JudgeRecord(Record):
    used: bool
    reason: str
    winner_id: str | None = None
    adapter: str | None = None
    comparisons: list[JudgeComparison] = Field(default_factory=list)


class Run(Record):
    id: str
    created_at: str
    mode: Literal["fixture", "live"]
    question: str
    task: Task
    source_plan: SourcePlan
    evidence: list[Passage]
    candidates: list[Candidate]
    checks: dict[str, list[Check]]
    judgments: list[Judgment]
    selected_id: str | None
    tied_ids: list[str]
    final_text: str
    final_hash: str
    # v1 adds meaning-checked answers: every sentence supported, or some sentences unconfirmed.
    outcome: Literal["source_quotes", "abstained", "verified_answer", "partly_verified"]
    uncertainty: str
    trace: list[Trace]
    estimated_cost_usd: float = 0
    cost_complete: bool = True
    verifier: str | None = None
    judge: JudgeRecord | None = None


class FixtureReply(Record):
    id: str
    text: str = ""
    status: Literal["completed", "failed", "refused", "truncated"] = "completed"
    delay_seconds: float = Field(default=0, ge=0, le=10)


class Case(Record):
    question: str = Field(min_length=1, max_length=4000)
    replies: list[FixtureReply] = Field(min_length=1, max_length=5)

    @field_validator("replies")
    @classmethod
    def unique_ids(cls, replies):
        if len({r.id for r in replies}) != len(replies):
            raise ValueError("Each fixture needs a distinct ID")
        return replies


class ProviderConfig(Record):
    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1, max_length=100)
    input_usd_per_million: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    output_usd_per_million: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class Limits(Record):
    max_calls: int = Field(default=2, ge=1, le=5)
    call_timeout_seconds: float = Field(default=20, gt=0, le=60, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=30, gt=0, le=120, allow_inf_nan=False)
    max_output_tokens: int = Field(default=1024, ge=16, le=4096)
    max_estimated_usd: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)


class LiveConfig(Record):
    # Listed in priority order; v1 uses the first max_calls entries whose API key is set.
    providers: list[ProviderConfig] = Field(min_length=1, max_length=5)
    limits: Limits = Field(default_factory=Limits)
    # Optional separate model for step 8 meaning checks (falls back to an available provider).
    verifier: ProviderConfig | None = None
    # Estimated US dollars for one web search in step 5. Live web search will not run without it.
    search_usd_per_call: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("providers")
    @classmethod
    def unique_providers(cls, providers):
        if len({(p.provider, p.model) for p in providers}) != len(providers):
            raise ValueError("Configure each provider model once")
        return providers
