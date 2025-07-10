from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


class BotMessage(BaseModel):
    """
    Domain model for an incoming chat message.
    """

    question: str
    answer: str


class ViolationType(Enum):
    """
    Types of content violations.
    """

    OFF_TOPIC = "off_topic"
    PII = "pii"
    AD = "ad"
    SAFETY = "safety"


class ViolationLevel(IntEnum):
    """
    Severity levels for violations.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Violation(BaseModel):
    """
    A single detected violation.
    """

    violation_type: str
    level: ViolationLevel


class ServiceCheckResult(BaseModel):
    """
    Result of a single service check (e.g. PII, Safety, AdFilter, OffTopic, Rewrite).
    """

    safe: bool
    score: float
    masked_answer: str
    question: str
    censored_entities: Optional[List[str]] = None
    error: Optional[str] = None


class FinalCheckResult(BaseModel):
    final_verdict_safe: bool
    violations: Optional[List[Violation]]
    # score: float
    masked_answer: str
    all_checks: Dict[str, ServiceCheckResult]


class LLMRewriteResult(BaseModel):
    answer: str


@dataclass(frozen=True)
class LLMRequest:
    """
    Request payload for invoking an LLM rewrite.
    """

    prompt: str  # template with placeholders like {answer}
    model: Optional[str] = None
    ollama_host: Optional[str] = None
    api_key: Optional[str] = None
