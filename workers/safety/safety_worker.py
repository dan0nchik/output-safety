import asyncio
from repositories.kafka_bus import KafkaEventBus
from repositories.safety_classifier import SafetyClassifierRepository
from entities.data import BotMessage, ServiceCheckResult
from config import settings


async def handle(message: BotMessage, headers: dict):
    if headers.get("check_type") != "safety":
        return
    print(message)
    # 1) run the PII adapter
    result: ServiceCheckResult = SafetyClassifierRepository().process(message)
    print(result)
    # 2) publish partial result back
    await bus.publish(
        topic="check-results",
        message=result,
        headers={
            "request_id": headers["request_id"],
            "check_type": "safety",
        },
    )


async def main():
    global bus
    bus = KafkaEventBus(brokers=settings.kafka_brokers)
    await bus.subscribe(
        topic="check-requests", group_id="safety-service", handler=handle
    )


if __name__ == "__main__":
    asyncio.run(main())
