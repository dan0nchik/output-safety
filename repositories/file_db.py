import os
from typing import Optional

from pymongo import MongoClient, collection
from use_cases.ports.db_connector import IDBRepository
from entities.data import FinalCheckResult


class MongoResultRepository(IDBRepository):
    """
    Репозиторий, сохраняющий FinalCheckResult в MongoDB.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: str = "app_db",
        collection_name: str = "results",
    ):
        """
        Подключается к MongoDB по URI (по умолчанию берётся из переменной окружения MONGO_URI),
        выбирает базу и коллекцию.
        """
        # URI вида "mongodb://user:pass@host:port/?authSource=admin"
        uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection: collection.Collection = self.db[collection_name]

    def save(self, request_id: str, final_result: FinalCheckResult) -> None:
        """
        Сохраняет или обновляет документ с ключом request_id.
        Поля модели FinalCheckResult будут сохранены как вложенное JSON-поле result.
        """
        doc = {
            "request_id": request_id,
            "result": final_result.dict(),
        }
        # Если запись с таким request_id уже есть — обновляем, иначе вставляем новую
        self.collection.update_one(
            {"request_id": request_id},
            {"$set": doc},
            upsert=True,
        )

final_check_result = FinalCheckResult(safe=True, score=0.5, masked_answer="кайфы")
repo = MongoResultRepository()  # возьмёт MONGO_URI из окружения
repo.save("1", final_check_result)