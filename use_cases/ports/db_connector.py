from abc import abstractmethod, ABC
from entities.data import CheckResult


class IDBRepository(ABC):
    @abstractmethod
    async def client(self):
        pass
