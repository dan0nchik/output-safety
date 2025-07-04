from abc import abstractmethod, ABC


class IDBRepository(ABC):
    @abstractmethod
    async def client(self):
        pass
