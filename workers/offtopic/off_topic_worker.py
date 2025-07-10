import asyncio
from repositories.kafka_bus import KafkaEventBus
from repositories.off_topic_scorer import OffTopicRepository
from entities.data import BotMessage, ServiceCheckResult
from config import settings


async def handle(message: BotMessage, headers: dict):
    if headers.get("check_type") != "off_topic":
        return

    result: ServiceCheckResult = OffTopicRepository(
        settings.off_topic_model_name
    ).process(message)

    await bus.publish(
        topic="check-results",
        message=result,
        headers={
            "request_id": headers["request_id"],
            "check_type": "off_topic",
        },
    )


async def main():
    global bus
    bus = KafkaEventBus(brokers=settings.kafka_brokers)

    # subscribe as part of the "pii-service" group
    await bus.subscribe(
        topic="check-requests", group_id="off-topic-service", handler=handle
    )


if __name__ == "__main__":
    asyncio.run(main())
