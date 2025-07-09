import asyncio
from typing import Dict, List
from use_cases.ports.event_bus import EventBus, MessageHandler
from repositories.file_db import FileResultRepository
from use_cases.ports.db_connector import IDBRepository
from repositories.kafka_bus import KafkaEventBus
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
    Subscribes to partial check-results, gathers them per request_id,
    merges into a FinalCheckResult, and hands off persistence to an IResultRepository.
    """

    def __init__(
        self,
        event_bus: EventBus,
        repo: IDBRepository,
        checks: List[str] = ["pii", "safety", "ad", "off_topic"],
    ):
        self.bus = event_bus
        self.repo = repo
        self.checks = checks
        # In-memory buffer: request_id -> { check_type: ServiceCheckResult }
        self._pending: Dict[str, Dict[str, ServiceCheckResult]] = {}

    async def handle(self, message: ServiceCheckResult, headers: dict):
        request_id = headers.get("request_id")
        check_type = headers.get("check_type")
        if not request_id or not check_type:
            return

        bucket = self._pending.setdefault(request_id, {})
        bucket[check_type] = message

        # If we've collected all expected checks, merge & persist
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
        # pick the first non-empty masked_answer (e.g. from rewrite)
        masked = next((p.masked_answer for p in parts.values() if p.masked_answer), "")
        # collect violations for any failed part
        violations = []
        for ct, p in parts.items():
            if not p.safe:
                vt = getattr(ViolationType, ct.upper(), None)
                lvl = (
                    ViolationLevel.HIGH
                    if p.score > 0.8
                    else ViolationLevel.MEDIUM if p.score > 0.5 else ViolationLevel.LOW
                )
                violations.append(Violation(violation_type=vt, level=lvl))

        return FinalCheckResult(
            safe=safe,
            violations=violations,
            score=score,
            masked_answer=masked,
        )


async def main():
    # wire up the Kafka event bus
    bus = KafkaEventBus(brokers=settings.kafka_brokers)
    # choose your persistence adapter (file-based for now)
    repo = FileResultRepository(directory="results")
    # instantiate & start listening
    aggregator = AggregatorService(bus, repo)
    await bus.subscribe(
        topic="check-results",
        group_id="aggregator",
        handler=aggregator.handle,
    )


if __name__ == "__main__":
    asyncio.run(main())
