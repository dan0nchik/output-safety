from entities.data import CheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository


class LLMRewriteRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> CheckResult:
        # TODO
        pass


# Пример тестирования репозитория
msg = BotMessage("Question", "Answer")
service1 = LLMRewriteRepository()
result = service1.process(msg)
print(result)
