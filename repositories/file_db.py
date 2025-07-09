import os
from use_cases.ports.db_connector import IDBRepository
from entities.data import FinalCheckResult


class FileResultRepository(IDBRepository):
    """
    A simple file-based repository: writes each FinalCheckResult as JSON into a dedicated .txt
    file under a given directory.
    """

    def __init__(self, directory: str = "results"):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory

    def save(self, request_id: str, final_result: FinalCheckResult) -> None:
        # Each request_id gets its own file: <directory>/<request_id>.txt
        file_path = os.path.join(self.directory, f"{request_id}.txt")
        # Use BaseModel.json() to serialize
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_result.json())
