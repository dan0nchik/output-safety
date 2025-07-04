from entities.data import CheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository


class LLMRewriteRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> CheckResult:
        return CheckResult(True, [], 0, "", message.answer)


# Пример тестирования репозитория
msg = BotMessage("Question", "Answer")
service1 = LLMRewriteRepository()
result = service1.process(msg)
print(result)
