from entities.data import ServiceCheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository
from llm_off_topic import LLMOffTopic


class LLMOffTopicRepository(IMLServiceRepository):
    """Репозиторий для проверки релевантности ответа на вопрос.

    Использует LLM для оценки соответствия ответа заданному вопросу.
    Возвращает результат проверки с оценкой релевантности от 0 до 1.
    """

    def __init__(self):
        """Инициализирует экземпляр LLM для оценки релевантности."""
        self.llm = LLMOffTopic()

    def process(self, message: BotMessage) -> ServiceCheckResult:
        """Обрабатывает сообщение и проверяет релевантность ответа.

        Args:
            message: Объект BotMessage, содержащий вопрос и ответ для проверки.

        Returns:
            ServiceCheckResult: Результат проверки, содержащий:
                - is_safe: bool (True если ответ релевантен)
                - score: float (оценка релевантности от 0 до 1)
                - answer: str (оригинальный ответ)
                - error: Optional[str] (сообщение об ошибке)
        """
        prompt = (
            "Оцени, релевантен ли следующий ответ данному вопросу.\n"
            f"Вопрос: {message.question}\n"
            f"Ответ: {message.answer}\n\n"
            "Ответь строго в формате:\n"
            "is_relevant: да/нет\n"
            "score: <число от 0 до 1, означающее процент соответствия ответа вопросу>"
        )

        error, is_safe, score = None, False, 0.0

        try:
            response = self.llm.ask(prompt)
        except Exception as exc:
            error = f"Ошибка при запросе к LLM: {str(exc)}"

        if error is None:
            # Парсинг ответа LLM
            try:
                lines = response.lower().splitlines()
                is_safe = any("да" in line for line in lines if "is_relevant" in line)

                score_line = next(line for line in lines if "score" in line)
                score = float(score_line.split(":")[1].strip())

            except StopIteration:
                error = "LLM не вернул обязательные поля is_relevant и score"
            except ValueError:
                error = "Некорректный формат оценки score в ответе LLM"
            except Exception as exc:
                error = f"Неожиданная ошибка при обработке ответа LLM: {str(exc)}"

        return ServiceCheckResult(
            safe=is_safe, score=score, masked_answer=message.answer, error=error
        )
