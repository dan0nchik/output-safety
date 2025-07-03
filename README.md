# output-safety

Сервис для проверки контента

## Установка

Требования:

- Python 3.9
- Форматтер кода [Ruff](https://docs.astral.sh/ruff/)
- Docker

Создаем виртуальное окружение

```bash
python3 -m venv venv
```

Переходим в него

```bash
source venv/bin/activate
```

Устанавливаем зависимости (пока там пусто, все зависимости типа pandas и тд пишите в этот файл)

```bash
pip install -r requirements.txt
```

Запускаем сервис. Пример вызова класса там же

```bash
python3 ./repositories/llm_rewrite.py
```

# Правила разработки

- Модули и файлы — `snake_case`.
- Классы — `PascalCase`.
- Константы — `UPPER_SNAKE_CASE`.
- Каждую фичу пишем в своей ветке (пример: `feature/llm_rewrite`).
- Перед комитом прогоняем Ruff форматирование
- Текст коммита на английском
- Название коммита: для фичи — `feat: <название>`, для багфикса — `fix: <название>`.

## Структура проекта

## Описание слоёв

### (от конкретного к более абстрактному)

### 1. `entities/`

- **Содержит чистые модели** (`@dataclass`):
    - `BotMessage`
    - `CheckResult`
    - `Violation` и др.
- **Не зависит** ни от БД, ни от API, ни от ML.

### 2. `use_cases/ports/`

- **Интерфейсы** (абстракции, «порты») для внешних систем:
    - `DBConnector` (логирование, хранение)
    - `QueueManager` (Kafka/Redis)
    - `MLServiceGateway` (вызовы PII, Safety, Ad-filter, Off-topic, Rewrite)
- **Не содержит реализации**, только методы `abstractmethod`.

### 3. `use_cases/check_message.py`

- **Главный сценарий** (Decision Engine):
    1. Отправляет в конструктор пять портов (PII, Safety, Ad, OffTopic, LLM-rewrite).
    2. Вызывает каждый из них последовательно или параллельно.
    3. Собирает и агрегирует результаты в единый `CheckResult`.
    4. Применяет рекурсивное переписывание (rewrite) при необходимости.

### 4. `repositories/`

- **Конкретные адаптеры** для каждого порта:
    - Вызовы ML-сервисов (BERT, NER, TF-IDF, эмбеддинги)
    - БД-коннектор (Postgres, SQLite…)
    - Очередь сообщений (Kafka, RabbitMQ…)
- **Основная логика находится здесь**

### 5. `presentation/api.py` (TODO)

- **Точка входа в приложение**:
    1. Загружает конфиг из `.env`.
    2. Составляет DI-контейнер: какие адаптеры впрыскивать в Use Case.
    3. Поднимает HTTP-сервер (FastAPI/Flask) или CLI, обрабатывает входящие запросы, вызывает
       `CheckMessageUseCase.execute()` и возвращает JSON.

---