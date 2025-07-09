import asyncio
from repositories.kafka_bus import KafkaEventBus
from repositories.pii_detector import PIIDetectorRepository
from entities.data import BotMessage, ServiceCheckResult
from config import settings


async def handle(message: BotMessage, headers: dict):
    if headers.get("check_type") != "pii":
        return

    # 1) run the PII adapter
    result: ServiceCheckResult = PIIDetectorRepository().process(message)

    # 2) publish partial result back
    await bus.publish(
        topic="check-results",
        message=result,
        headers={
            "request_id": headers["request_id"],
            "check_type": "pii",
        },
    )


async def main():
    global bus
    bus = KafkaEventBus(brokers=settings.kafka_brokers)
    await bus.subscribe(topic="check-requests", group_id="pii-service", handler=handle)


if __name__ == "__main__":
    asyncio.run(main())
