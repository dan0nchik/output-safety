# workers/aggregator/aggregator.py

import asyncio
import re
from typing import Dict, List

from use_cases.ports.event_bus import EventBus, MessageHandler
from use_cases.ports.db_connector import IDBRepository
from use_cases.ports.ml_service import ILLMRewriteRepository
from repositories.kafka_bus import KafkaEventBus
from repositories.file_db import MongoResultRepository
from repositories.llm_rewrite import GigachatRewriteRepository
from config import settings
from entities.data import (
    ServiceCheckResult,
    FinalCheckResult,
    Violation,
    ViolationType,
    ViolationLevel,
    LLMRequest,
    LLMRewriteResult,
)


class AggregatorService:
    """
    Gathers partial ServiceCheckResults per request_id, merges them
    into a FinalCheckResult, and hands off persistence to an IDBRepository.
    """

    def __init__(
        self,
        repo: IDBRepository,
        rewriter: ILLMRewriteRepository,
        checks: List[str] = ["pii", "safety", "ad", "off_topic"],
    ):
        self.repo = repo
        self.rewriter = rewriter
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
    def _strip_masked_words(text: str) -> str:
        def is_masked(word: str) -> bool:
            if len(word) < 2:
                return False
            if len(set(word)) == 1 and not word.isalnum():
                return True
            if word.upper() in {"[MASK]", "[REDACTED]"}:
                return True
            if re.fullmatch(r"[^a-zA-Z0-9]+", word) and len(word) > 3:
                return True
            return False

        return " ".join(w for w in text.split() if not is_masked(w))

    def _rewrite(self, masked_text: str, problems: list[str]) -> str:
        """
        Use the LLM to rewrite the text, handling any exceptions.
        Returns the rewritten text or a placeholder if it fails.
        """
        problem_map = {"ad": "реклама", "off_topic": "несоответсвие теме"}
        problems_str = ", ".join(problem_map[p] for p in problems if p in problem_map)
        prompt = (
            f"В данном тексте (в тэге <TEXT>) следующие проблемы: {problems_str}. "
            f"Перепиши текст, исправив ошибки. Верни результат в тэге <RES>. "
            f"<TEXT>{masked_text}</TEXT>"
        )
        try:
            llm_request: LLMRequest = LLMRequest(
                prompt=prompt,
                api_key=settings.gigachat_api,
            )
            response: LLMRewriteResult = self.rewriter.process(llm_request)
            return response.answer if response else "[REWRITE_NEEDED]"
        except Exception as exc:
            print(f"[aggregator] Rewrite error: {exc}")
            return "[REWRITE_NEEDED]"

    def _merge(self, parts: Dict[str, ServiceCheckResult]) -> FinalCheckResult:
        final_safe = all(p.safe for p in parts.values())
        violations: List[Violation] = []

        ad_or_offtopic_unsafe = False
        failed_checks = []

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
                    failed_checks.append(check_type)

        # Get any available answer (they all should use same original base)
        base_answer = next(
            (p.masked_answer for p in parts.values() if p.masked_answer), ""
        )

        # Handle PII or SAFETY violations
        pii_result = parts.get("pii")
        safety_result = parts.get("safety")
        pii_or_safety_failed = (pii_result and not pii_result.safe) or (
            safety_result and not safety_result.safe
        )

        if pii_or_safety_failed:
            cleaned = self._strip_masked_words(base_answer)
            if ad_or_offtopic_unsafe:
                rewritten = self._rewrite(cleaned, failed_checks)
                return FinalCheckResult(
                    final_verdict_safe=final_safe,
                    violations=violations,
                    masked_answer=rewritten or "[REWRITE_NEEDED]",
                    all_checks=parts,
                )
            else:
                return FinalCheckResult(
                    final_verdict_safe=final_safe,
                    violations=violations,
                    masked_answer=cleaned,
                    all_checks=parts,
                )

        # No PII/SAFETY issues, but AD or OFF_TOPIC fail
        if ad_or_offtopic_unsafe:
            rewritten = self._rewrite(base_answer, failed_checks)
            return FinalCheckResult(
                final_verdict_safe=final_safe,
                violations=violations,
                masked_answer=rewritten or "[REWRITE_NEEDED]",
                all_checks=parts,
            )

        # Everything safe
        return FinalCheckResult(
            final_verdict_safe=final_safe,
            violations=violations,
            masked_answer=base_answer,
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

    rewriter: GigachatRewriteRepository = GigachatRewriteRepository()

    # 3) instantiate service
    global aggregator
    aggregator = AggregatorService(repo, rewriter)

    # 4) subscribe
    await bus.subscribe(
        topic="check-results",
        group_id="aggregator",
        handler=_raw_handler,  # we wrap to deserialize correctly
    )


if __name__ == "__main__":
    asyncio.run(main())
