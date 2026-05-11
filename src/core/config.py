import functools
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index_prefix: str = "highwatch"
    anthropic_api_key: str = ""
    gdrive_credentials_path: str = "credentials.json"
    gdrive_folder_id: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    llm_model: str = "claude-sonnet-4-5"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    rrf_k: int = 60
    llm_provider: Literal["anthropic", "google"] = "anthropic"
    google_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
