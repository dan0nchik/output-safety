from entities.data import ServiceCheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository


class PIIDetectorRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> ServiceCheckResult:
        # TODO
        pass
