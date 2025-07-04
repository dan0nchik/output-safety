from entities.data import CheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository
from sentence_transformers import SentenceTransformer, util


class OffTopicRepository(IMLServiceRepository):
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-albert-small-v2") # легкая модель для эмбеддингов

    def process(self, message: BotMessage) -> CheckResult:
        str1 = message.question
        str2 = message.answer

        # Получение эмбеддингов
        embedding1 = self.model.encode(str1, convert_to_tensor=True)
        embedding2 = self.model.encode(str2, convert_to_tensor=True)

        # Косинусное сходство
        score = util.cos_sim(embedding1, embedding2).item()

        is_safe = score >= 0.5

        return CheckResult(is_safe, [], score, "", message.answer)


# Пример тестирования репозитория (чем больше к 1 ответ, тем лучше)
str1 = "Let's talk about London."
str2 = "London is the capital of Great Britain."
msg = BotMessage(str1, str2)
service1 = OffTopicRepository()
result = service1.process(msg)
print(result)
