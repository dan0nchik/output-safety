from abc import abstractmethod, ABC
from entities.data import ServiceCheckResult, BotMessage, LLMRequest, LLMRewriteResult


class IMLServiceRepository(ABC):
    @abstractmethod
    async def process(self, message: BotMessage) -> ServiceCheckResult:
        pass


class ILLMRewriteRepository(ABC):
    @abstractmethod
    async def process(self, request: LLMRequest) -> LLMRewriteResult:
        pass
