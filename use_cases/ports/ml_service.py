from abc import abstractmethod, ABC
from entities.data import ServiceCheckResult, BotMessage, LLMRequest


class IMLServiceRepository(ABC):
    @abstractmethod
    async def process(self, message: BotMessage) -> ServiceCheckResult:
        pass


class ILLMRewriteRepository(ABC):
    @abstractmethod
    async def process(
        self, message: BotMessage, request: LLMRequest
    ) -> ServiceCheckResult:
        pass
