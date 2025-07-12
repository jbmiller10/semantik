# shared/config/base.py

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """
    Base configuration shared by all services.
    This contains settings that are common across vecpipe and webui.
    """

    # Project root directory, calculated automatically
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent.resolve()

    # Environment Configuration
    ENVIRONMENT: str = "development"  # Options: development, production

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    DEFAULT_COLLECTION: str = "work_docs"

    # Internal API Configuration
    INTERNAL_API_KEY: str | None = None

    # Data & Processing Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # For Docker environments, use the mounted volume path
    @property
    def data_dir(self) -> Path:
        docker_data = Path("/app/data")
        return docker_data if docker_data.exists() else self.DATA_DIR

    @property
    def logs_dir(self) -> Path:
        docker_logs = Path("/app/logs")
        return docker_logs if docker_logs.exists() else self.LOGS_DIR

    # Common paths using the dynamic data_dir
    @property
    def file_tracking_db(self) -> Path:
        return self.data_dir / "file_tracking.json"

    @property
    def webui_db(self) -> Path:
        return self.data_dir / "webui.db"

    @property
    def extract_dir(self) -> Path:
        return self.data_dir / "extract"

    @property
    def ingest_dir(self) -> Path:
        return self.data_dir / "ingest"

    @property
    def loaded_dir(self) -> Path:
        return self.data_dir / "loaded"

    @property
    def reject_dir(self) -> Path:
        return self.data_dir / "rejects"

    @property
    def manifest_file(self) -> Path:
        return self.data_dir / "filelist.null"

    @property
    def error_log(self) -> Path:
        return self.logs_dir / "error_extract.log"

    @property
    def cleanup_log(self) -> Path:
        return self.logs_dir / "cleanup.log"

    # Pydantic model config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Create data/log directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
