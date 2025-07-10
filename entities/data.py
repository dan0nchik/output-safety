from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BotMessage:
    question: str
    answer: str


@dataclass
class ViolationLevel:
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class ViolationType:
    OFF_TOPIC = "off_topic"
    PII = "pii"
    AD = "ad"
    SAFETY = "safety"


@dataclass(frozen=True)
class LLMRequest:
    prompt: str  # переменные в промпте всавлять в виде {variable_name}
    model: Optional[str]
    ollama_host: Optional[str]
    api_key: Optional[str]


@dataclass
class Violation:
    violation_type: ViolationType
    level: ViolationLevel


# result per service
@dataclass
class ServiceCheckResult:
    safe: bool
    score: int
    masked_answer: str
    censored_entities: Optional[List[str]] = None


# final result after decision engine
@dataclass
class FinalCheckResult:
    safe: bool
    violations: List[Violation]
    score: int
    masked_answer: str
