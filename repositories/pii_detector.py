from entities.data import CheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository


class PIIDetectorRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> CheckResult:
        # TODO
        pass
