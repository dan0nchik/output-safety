import os
from typing import Optional
from pymongo import MongoClient, collection
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError
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
        uri = mongo_uri or os.getenv("MONGO_URI")
        print(uri)
        if not uri:
            raise ValueError(
                "Mongo URI must be provided via argument or MONGO_URI env var"
            )

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Force connection to check auth
        except ServerSelectionTimeoutError as e:
            raise ConnectionError("Could not connect to MongoDB") from e
        except OperationFailure as e:
            raise PermissionError("MongoDB authentication failed") from e

        self.db = self.client[db_name]
        self.collection: collection.Collection = self.db[collection_name]

    def save(self, request_id: str, final_result: FinalCheckResult) -> None:
        """
        Сохраняет или обновляет документ с ключом request_id.
        Поля модели FinalCheckResult будут сохранены как вложенное JSON-поле result.
        """
        doc = {
            "request_id": request_id,
            "result": final_result.model_dump(),
        }

        try:
            self.collection.update_one(
                {"request_id": request_id},
                {"$set": doc},
                upsert=True,
            )
            print(f"[mongo] Saved result for request_id={request_id}")
        except OperationFailure as e:
            raise RuntimeError("Failed to write to MongoDB") from e
