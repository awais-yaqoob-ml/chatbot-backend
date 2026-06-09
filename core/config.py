from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ======================================================
    # GROQ
    # ======================================================

    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
    )

    groq_model: str = Field(
        default="llama3-70b-8192",
        alias="GROQ_MODEL",
    )

    # ======================================================
    # EMBEDDINGS
    # ======================================================

    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBED_MODEL",
    )

    # ======================================================
    # WEAVIATE
    # ======================================================

    # ======================================================
    # WEAVIATE
    # ======================================================

    weaviate_host: str = Field(
        default="localhost",
        alias="WEAVIATE_HOST",
    )

    weaviate_port: int = Field(
        default=8080,
        alias="WEAVIATE_PORT",
    )

    weaviate_grpc_port: int = Field(
        default=50051,
        alias="WEAVIATE_GRPC_PORT",
    )

    # ======================================================
    # STORAGE
    # ======================================================

    assets_dir: str = Field(
        default="./extracted_assets",
        alias="ASSETS_DIR",
    )

    # ======================================================
    # RETRIEVAL
    # ======================================================

    top_k: int = Field(
        default=5,
        alias="TOP_K",
    )

    retrieval_threshold: float = Field(
        default=0.50,
        alias="RETRIEVAL_THRESHOLD",
    )

    # ======================================================
    # API
    # ======================================================

    api_v1_prefix: str = Field(
        default="/api/v1",
        alias="API_V1_PREFIX",
    )

    # ======================================================
    # LOGGING
    # ======================================================

    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    log_dir: str = Field(
        default="./logs",
        alias="LOG_DIR",
    )

    # ======================================================
    # FILES
    # ======================================================

    max_upload_size_mb: int = Field(
        default=20,
        alias="MAX_UPLOAD_SIZE_MB",
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def assets_path(self) -> Path:
        return Path(self.assets_dir)


    @property
    def logs_path(self) -> Path:
        return Path(self.log_dir)


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    """
    return Settings()


settings = get_settings()