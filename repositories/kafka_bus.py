# infrastructure/adapters/kafka_bus.py
import json
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from use_cases.ports.event_bus import EventBus, MessageHandler
from entities.data import BotMessage


class KafkaEventBus(EventBus):
    def __init__(self, brokers: str):
        self.brokers = brokers
        self._producer = None

    async def _get_producer(self) -> AIOKafkaProducer:
        if self._producer is None:
            p = AIOKafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda m: json.dumps(m).encode(),
            )
            await p.start()
            self._producer = p
        return self._producer

    async def publish(self, topic: str, message: BotMessage, headers: dict) -> None:
        producer = await self._get_producer()
        await producer.send_and_wait(
            topic,
            message.dict(),
            headers=[(k, str(v).encode()) for k, v in headers.items()],
        )

    async def subscribe(
        self, topic: str, group_id: str, handler: MessageHandler
    ) -> None:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.brokers,
            group_id=group_id,
            value_deserializer=lambda b: json.loads(b.decode()),
        )
        await consumer.start()
        try:
            async for record in consumer:
                msg = BotMessage(**record.value)
                hdrs = {k: v.decode() for k, v in record.headers or []}
                await handler(msg, hdrs)
        finally:
            await consumer.stop()
