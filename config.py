from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    # LLM
    ollama_base_url: str = Field("", alias="LLM_BASE_URL")
    ollama_model_name: str = Field("", alias="LLM_MODEL_NAME")
    ollama_prompt: str = Field("перепиши текст", alias="LLM_PROMPT")
    off_topic_model_name: str = Field(
        "paraphrase-albert-small-v2", alias="OFF_TOPIC_MODEL_NAME"
    )


settings = Settings()
