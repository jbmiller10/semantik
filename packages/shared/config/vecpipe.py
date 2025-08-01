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

    # Adaptive Batch Size Configuration
    ENABLE_ADAPTIVE_BATCH_SIZE: bool = True  # Enables dynamic batch sizing based on GPU memory availability
    MIN_BATCH_SIZE: int = 1  # Minimum batch size for OOM recovery
    MAX_BATCH_SIZE: int = 256  # Maximum allowed batch size
    BATCH_SIZE_SAFETY_MARGIN: float = 0.2  # 20% safety margin for GPU memory to prevent OOM
    BATCH_SIZE_INCREASE_THRESHOLD: int = 10  # Number of successful batches before attempting to increase size

    # Additional Paths specific to vecpipe
    @property
    def operations_dir(self) -> Path:
        return self.data_dir / "operations"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"
