import os
import requests
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()


class LLMOffTopic:
    """Класс для взаимодействия с LLM через API OpenRouter.

    Позволяет отправлять запросы к языковой модели и получать ответы.
    Использует API ключ из переменных окружения.
    """

    def __init__(self):
        """Инициализация клиента OpenRouter API."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "deepseek/deepseek-chat-v3-0324:free"  # Модель по умолчанию
        self.url = "https://openrouter.ai/api/v1/chat/completions"  # URL API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def ask(self, prompt: str) -> str:
        """Отправляет запрос к языковой модели и возвращает ответ.

        Args:
            prompt: Текст запроса для языковой модели

        Returns:
            str: Текст ответа от языковой модели

        Raises:
            RuntimeError: В случае ошибки при обращении к API
        """
        # Формирование тела запроса
        body = {
            "model": self.model,  # Используемая модель
            "messages": [
                {
                    "role": "system",
                    "content": "Ты эксперт по анализу соответствия ответа и вопроса."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,  # Параметр для детерминированных ответов
        }

        # Отправка POST-запроса к API
        response = requests.post(self.url, headers=self.headers, json=body)

        # Проверка статуса ответа
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {response.status_code}: {response.text}"
            )

        # Извлечение и возврат текста ответа
        return response.json()["choices"][0]["message"]["content"].strip()