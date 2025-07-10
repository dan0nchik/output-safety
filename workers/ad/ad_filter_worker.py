# workers/ad/ad_filter_worker.py

import asyncio
from use_cases.ports.event_bus import EventBus, MessageHandler
from repositories.kafka_bus import KafkaEventBus
from repositories.ad_filter import AdFilterRepository
from entities.data import BotMessage, ServiceCheckResult
from config import settings


async def handle_ad(message: BotMessage, headers: dict):
    # only process "ad" check_type
    if headers.get("check_type") != "ad":
        return

    # run the AdFilter adapter
    result: ServiceCheckResult = AdFilterRepository(
        settings.ad_filter_model_name
    ).process(message)

    # publish the partial result back to Kafka
    await bus.publish(
        topic="check-results",
        message=result,  # ServiceCheckResult is a BaseModel → .dict() under the hood
        headers={
            "request_id": headers["request_id"],
            "check_type": "ad",
        },
    )
    print(f"[ad_worker] Emitted AD result for {headers['request_id']}")


async def main():
    global bus  # shared by handler
    # inject the Kafka adapter
    bus = KafkaEventBus(brokers=settings.kafka_brokers)

    # subscribe to scatter topic as part of the "ad-service" group
    await bus.subscribe(
        topic="check-requests",
        group_id="ad-service",
        handler=handle_ad,  # type: MessageHandler
    )


if __name__ == "__main__":
    asyncio.run(main())
