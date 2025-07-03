from dataclasses import dataclass
from typing import List


@dataclass
class BotMessage:
    answer: str


class ViolationLevel:
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class Violation:
    violation_type: str
    level: ViolationLevel
    message: str


@dataclass
class CheckResult:
    safe: bool
    violations: List[Violation]
    score: int
    actions: str
    masked_answer: str
