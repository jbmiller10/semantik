# shared/config/vecpipe.py

from pathlib import Path

from .base import BaseConfig


class VecpipeConfig(BaseConfig):
    """
    Vecpipe-specific configuration.
    Contains settings specific to the search engine and embedding pipeline.
    """

    # Embedding Model Configuration
    USE_MOCK_EMBEDDINGS: bool = False
    DEFAULT_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_QUANTIZATION: str = "float16"

    # Service Ports
    SEARCH_API_PORT: int = 8000
    METRICS_PORT: int = 9091

    # Service URLs (for internal API calls)
    # Support Docker service names through environment variables
    SEARCH_API_URL: str = "http://localhost:8000"

    # Model Management
    MODEL_UNLOAD_AFTER_SECONDS: int = 300  # 5 minutes default

    # Additional Paths specific to vecpipe
    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"
