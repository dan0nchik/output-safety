# workers/aggregator/aggregator.py

import asyncio
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

    def _merge(self, parts: Dict[str, ServiceCheckResult]) -> FinalCheckResult:
        # overall safety = all parts safe
        safe = all(p.safe for p in parts.values())
        # take the highest score
        score = max(p.score for p in parts.values())
        # pick the first non-empty masked_answer
        masked = next((p.masked_answer for p in parts.values() if p.masked_answer), "")
        # collect violations for any failed part
        violations: List[Violation] = []
        for check_type, p in parts.items():
            if not p.safe:
                vt = getattr(ViolationType, check_type.upper(), None)
                lvl = (
                    ViolationLevel.HIGH
                    if p.score > 0.8
                    else ViolationLevel.MEDIUM
                    if p.score > 0.5
                    else ViolationLevel.LOW
                )
                violations.append(Violation(violation_type=vt, level=lvl))

        return FinalCheckResult(
            safe=safe,
            violations=violations,
            score=score,
            masked_answer=masked,
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
