from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = Field(..., env="LLM_BASE_URL")
    ollama_model_name: str = Field(..., env="LLM_MODEL_NAME")
    ollama_prompt: str = Field("перепиши текст", env="LLM_PROMPT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
