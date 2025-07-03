from entities.data import CheckResult
from use_cases.ports.ml_service import IMLServiceRepository


class AdFilterRepository(IMLServiceRepository):
    def process(self, text: str) -> CheckResult:
        # TODO
        pass
