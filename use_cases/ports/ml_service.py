from abc import abstractmethod, ABC
from entities.data import CheckResult


class IMLServiceRepository(ABC):
    @abstractmethod
    async def process(self, text: str) -> CheckResult:
        pass
