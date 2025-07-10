from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable
from entities.data import BotMessage

MessageHandler = Callable[[BotMessage, dict], Awaitable[Any]]


class EventBus(ABC):
    @abstractmethod
    async def publish(self, topic: str, message: BotMessage, headers: dict) -> None: ...

    @abstractmethod
    async def subscribe(
        self, topic: str, group_id: str, handler: MessageHandler
    ) -> None: ...
