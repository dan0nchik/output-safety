from abc import ABC, abstractmethod


class IQueueRepository(ABC):
    @abstractmethod
    async def client(self):
        pass
