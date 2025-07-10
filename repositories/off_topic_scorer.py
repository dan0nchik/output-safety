from entities.data import ServiceCheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository
from sentence_transformers import SentenceTransformer, util


class OffTopicRepository(IMLServiceRepository):
    """Репозиторий для проверки релевантности ответа на вопрос.

    Использует модель Sentence Transformers для сравнения эмбеддингов текстов.
    Возвращает результат проверки с метрикой косинусного сходства.

    Attributes:
        model: Модель для генерации текстовых эмбеддингов.
    """

    def __init__(self, model_name: str):
        """Инициализирует модель для генерации эмбеддингов."""
        self.model_name = model_name
        self.model = SentenceTransformer(
            self.model_name,
        )  # Легкая модель для эмбеддингов

    def process(self, message: BotMessage) -> ServiceCheckResult:
        """Обрабатывает сообщение и проверяет релевантность ответа.

        Args:
            message: Объект сообщения с вопросом и ответом для проверки.

        Returns:
            ServiceCheckResult: Результат проверки, содержащий:
                - Флаг is_safe (True если ответ релевантен)
                - Значение косинусного сходства
                - Исходный ответ
        """
        str1 = message.question
        str2 = message.answer

        # Генерация векторных представлений текстов
        embedding1 = self.model.encode(str1, convert_to_tensor=True)
        embedding2 = self.model.encode(str2, convert_to_tensor=True)

        # Вычисление метрики косинусного сходства между векторами
        score = util.cos_sim(embedding1, embedding2).item()

        # Пороговое значение для определения релевантности (0.5 - примерное значение)
        is_safe = score >= 0.5

        return ServiceCheckResult(
            safe=is_safe,
            score=score,
            masked_answer=message.answer,
            question=message.question,
        )
