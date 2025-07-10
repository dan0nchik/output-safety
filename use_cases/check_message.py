# use_cases/check_message.py
from uuid import uuid4
from entities.data import BotMessage
from use_cases.ports.event_bus import EventBus


class CheckMessageUseCase:
    """
    Instead of running all checks here, we simply
    publish one message per check_type into Kafka.
    """

    def __init__(
        self,
        event_bus: EventBus,
        request_topic: str = "check-requests",
        checks: list[str] = ["pii", "safety", "ad", "off_topic"],
    ):
        self.event_bus = event_bus
        self.request_topic = request_topic
        self.checks = checks

    async def enqueue(self, message: BotMessage) -> str:
        # Generate a unique correlation ID
        request_id = str(uuid4())
        # Scatter: publish one message PER check
        for check_type in self.checks:
            await self.event_bus.publish(
                topic=self.request_topic,
                message=message,
                headers={"request_id": request_id, "check_type": check_type},
            )
        return request_id
