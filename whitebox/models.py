from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Run:
    """A single model run within a decision."""

    model: str
    answer: str
    logprob: float
    latency_ms: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        return cls(
            model=data.get("model", ""),
            answer=data.get("answer", ""),
            logprob=data.get("logprob", 0.0),
            latency_ms=data.get("latency_ms", 0),
        )


@dataclass
class Decision:
    """Represents a single classification decision."""

    id: str
    status: str
    value: Optional[str] = None
    confidence: Optional[float] = None
    verdict: Optional[str] = None
    escalated: bool = False
    runs: list[Run] = field(default_factory=list)
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    created_at: Optional[str] = None
    mode: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        runs_data = data.get("runs") or []
        return cls(
            id=data.get("id", ""),
            status=data.get("status", ""),
            value=data.get("value"),
            confidence=data.get("confidence"),
            verdict=data.get("verdict"),
            escalated=data.get("escalated", False),
            runs=[Run.from_dict(r) for r in runs_data],
            latency_ms=data.get("latency_ms"),
            cost_usd=data.get("cost_usd"),
            created_at=data.get("created_at"),
            mode=data.get("mode"),
        )


@dataclass
class Batch:
    """Represents a bulk decision batch."""

    id: str
    status: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    progress: float = 0.0
    webhook_url: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Batch:
        return cls(
            id=data.get("id", ""),
            status=data.get("status", ""),
            total=data.get("total", 0),
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
            progress=data.get("progress", 0.0),
            webhook_url=data.get("webhook_url"),
            completed_at=data.get("completed_at"),
            created_at=data.get("created_at"),
        )


@dataclass
class Review:
    """Represents a human review request for an escalated decision."""

    id: int
    decision_id: str
    status: str
    input: Optional[str] = None
    options: Optional[list[str]] = None
    model_votes: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    sla_deadline: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Review:
        return cls(
            id=data.get("id", 0),
            decision_id=data.get("decision_id", ""),
            status=data.get("status", ""),
            input=data.get("input"),
            options=data.get("options"),
            model_votes=data.get("model_votes"),
            confidence=data.get("confidence"),
            sla_deadline=data.get("sla_deadline"),
            created_at=data.get("created_at"),
        )
