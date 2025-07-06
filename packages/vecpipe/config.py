# vecpipe/config.py

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Settings are loaded from a .env file or environment variables.
    """

    # Project root directory, calculated automatically
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

    # Qdrant Configuration
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    DEFAULT_COLLECTION: str = "work_docs"

    # Embedding Model Configuration
    USE_MOCK_EMBEDDINGS: bool = False
    DEFAULT_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_QUANTIZATION: str = "float16"

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"

    # Data & Processing Paths
    # Use paths relative to the project root for portability
    FILE_TRACKING_DB: Path = PROJECT_ROOT / "data" / "file_tracking.json"
    WEBUI_DB: Path = PROJECT_ROOT / "data" / "webui.db"
    EXTRACT_DIR: Path = PROJECT_ROOT / "data" / "extract"
    INGEST_DIR: Path = PROJECT_ROOT / "data" / "ingest"
    LOADED_DIR: Path = PROJECT_ROOT / "data" / "loaded"
    REJECT_DIR: Path = PROJECT_ROOT / "data" / "rejects"
    MANIFEST_FILE: Path = PROJECT_ROOT / "data" / "filelist.null"

    # Logging
    ERROR_LOG: Path = PROJECT_ROOT / "logs" / "error_extract.log"
    CLEANUP_LOG: Path = PROJECT_ROOT / "logs" / "cleanup.log"

    # Service Ports
    SEARCH_API_PORT: int = 8000
    WEBUI_PORT: int = 8080

    # Service URLs (for internal API calls)
    SEARCH_API_URL: str = "http://localhost:8000"
    WEBUI_URL: str = "http://localhost:8080"

    # Additional Paths
    JOBS_DIR: Path = PROJECT_ROOT / "data" / "jobs"
    OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "output"

    # Pydantic model config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Instantiate settings once and export
settings = Settings(
    QDRANT_HOST=os.getenv("QDRANT_HOST", "localhost"), JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY", "default-secret-key")
)

# Create data/log directories if they don't exist
(settings.PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
(settings.PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
