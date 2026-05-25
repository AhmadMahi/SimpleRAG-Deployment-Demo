from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pinecone_api_key: str = ""
    pinecone_index: str = "basic-rag-index"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    upload_dir: str = "uploads"
    state_file: str = "data/state.json"
    cache_size: int = 40
    conversation_history_size: int = 60

    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "basic-rag"

    allow_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
