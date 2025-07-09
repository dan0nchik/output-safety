from abc import ABC, abstractmethod
from entities.data import FinalCheckResult


class IDBRepository(ABC):
    """
    Port for persisting final check results. Can be implemented for files, databases, etc.
    """

    @abstractmethod
    def save(self, request_id: str, final_result: FinalCheckResult) -> None:
        """
        Persist the FinalCheckResult for the given request_id.
        """
        ...
