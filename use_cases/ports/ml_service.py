from abc import abstractmethod, ABC
from entities.data import CheckResult, BotMessage


class IMLServiceRepository(ABC):
    @abstractmethod
    async def process(self, message: BotMessage) -> CheckResult:
        pass
