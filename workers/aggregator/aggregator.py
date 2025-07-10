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
        final_safe = all(p.safe for p in parts.values())

        # Merge all masked answers by overlaying masks over the original
        # Start with the unmasked version from a 'safe' result if available
        base_answer = next((p.masked_answer for p in parts.values() if p.safe), "")
        if not base_answer:
            # fallback: use any masked answer
            base_answer = next(
                (p.masked_answer for p in parts.values() if p.masked_answer), ""
            )

        # Create a character-wise mask over base_answer
        composite = list(base_answer)
        for part in parts.values():
            for i, (orig_c, new_c) in enumerate(zip(base_answer, part.masked_answer)):
                if new_c != orig_c:
                    composite[i] = new_c

        unified_masked_answer = "".join(composite)

        violations: List[Violation] = []
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
