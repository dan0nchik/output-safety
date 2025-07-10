# workers/aggregator/aggregator.py

import asyncio
import re
from typing import Dict, List

from use_cases.ports.event_bus import EventBus, MessageHandler
from use_cases.ports.db_connector import IDBRepository
from repositories.kafka_bus import KafkaEventBus
from repositories.file_db import MongoResultRepository
from config import settings
from entities.data import (
    ServiceCheckResult,
    FinalCheckResult,
    Violation,
    ViolationType,
    ViolationLevel,
)


class AggregatorService:
    """
    Gathers partial ServiceCheckResults per request_id, merges them
    into a FinalCheckResult, and hands off persistence to an IDBRepository.
    """

    def __init__(
        self,
        repo: IDBRepository,
        checks: List[str] = ["pii", "safety", "ad", "off_topic"],
    ):
        self.repo = repo
        self.checks = checks
        # in-memory buffer: request_id -> { check_type: ServiceCheckResult }
        self._pending: Dict[str, Dict[str, ServiceCheckResult]] = {}

    async def handle(self, result: ServiceCheckResult, headers: dict):
        request_id = headers.get("request_id")
        check_type = headers.get("check_type")
        if not request_id or not check_type:
            return

        bucket = self._pending.setdefault(request_id, {})
        bucket[check_type] = result

        # once we've got every expected check, merge & persist
        if all(ct in bucket for ct in self.checks):
            parts = self._pending.pop(request_id)
            final = self._merge(parts)
            self.repo.save(request_id, final)
            print(f"[aggregator] Saved final result for {request_id}")

    @staticmethod
    def _is_masked_word(word: str) -> bool:
        if len(word) < 2:
            return False
        # Repeated single character (e.g. "****", "xxxxx", "XXX")
        if len(set(word)) == 1 and not word.isalnum():
            return True
        # Fully masked patterns (e.g. "********", "[MASK]", "[REDACTED]")
        if word.upper() in {"[MASK]", "[REDACTED]"}:
            return True
        # Long strings of non-letters/numbers
        if re.fullmatch(r"[^a-zA-Z0-9]+", word) and len(word) > 3:
            return True
        return False

    def _merge(self, parts: Dict[str, ServiceCheckResult]) -> FinalCheckResult:
        final_safe = all(p.safe for p in parts.values())

        violations: List[Violation] = []
        ad_or_offtopic_unsafe = False

        # Collect violations and see if ad or off_topic failed
        for check_type, result in parts.items():
            if not result.safe:
                vt = getattr(ViolationType, check_type.upper(), None)
                lvl = (
                    ViolationLevel.HIGH
                    if result.score > 0.8
                    else (
                        ViolationLevel.MEDIUM
                        if result.score > 0.5
                        else ViolationLevel.LOW
                    )
                )
                violations.append(Violation(violation_type=vt.value, level=lvl))
                if check_type in {"ad", "off_topic"}:
                    ad_or_offtopic_unsafe = True

        # Handle masking for PII and Safety
        pii_result = parts.get("pii")
        safety_result = parts.get("safety")

        if pii_result or safety_result:
            pii_words = pii_result.masked_answer.split() if pii_result else []
            safety_words = safety_result.masked_answer.split() if safety_result else []
            max_len = max(len(pii_words), len(safety_words))

            unified_words = []
            for i in range(max_len):
                pw = pii_words[i] if i < len(pii_words) else ""
                sw = safety_words[i] if i < len(safety_words) else ""

                if self._is_masked_word(pw) or self._is_masked_word(sw):
                    continue  # Drop censored word
                unified_words.append(pw or sw)

            unified_masked_answer = " ".join(unified_words)

        # If ad or off_topic were unsafe, force rewrite
        if ad_or_offtopic_unsafe:
            unified_masked_answer = "[REWRITE_NEEDED]"

        return FinalCheckResult(
            final_verdict_safe=final_safe,
            violations=violations,
            masked_answer=unified_masked_answer,
            all_checks=parts,
        )


# ———————— adapter + entrypoint —————————


async def _raw_handler(payload: dict, headers: dict):
    """
    KafkaEventBus will give us raw dicts here, so turn them
    into ServiceCheckResult before handing off to our service.
    """
    result = ServiceCheckResult(**payload)
    await aggregator.handle(result, headers)


async def main():
    # 1) wire up Kafka
    bus: EventBus = KafkaEventBus(brokers=settings.kafka_brokers)

    # 2) choose persistence adapter
    repo: IDBRepository = MongoResultRepository(mongo_uri=settings.mongo_uri)

    # 3) instantiate service
    global aggregator
    aggregator = AggregatorService(repo)

    # 4) subscribe
    await bus.subscribe(
        topic="check-results",
        group_id="aggregator",
        handler=_raw_handler,  # we wrap to deserialize correctly
    )


if __name__ == "__main__":
    asyncio.run(main())
