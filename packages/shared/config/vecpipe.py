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

    # GPU Memory Governor Settings
    ENABLE_MEMORY_GOVERNOR: bool = True  # Use GovernedModelManager with dynamic memory management
    GPU_MEMORY_MAX_PERCENT: float = 0.90  # Maximum GPU memory the application can use
    CPU_MEMORY_MAX_PERCENT: float = 0.50  # Maximum CPU memory for warm models
    ENABLE_CPU_OFFLOAD: bool = True  # Offload models to CPU instead of unloading
    EVICTION_IDLE_THRESHOLD_SECONDS: int = 120  # Idle time before model eligible for eviction
    PRESSURE_CHECK_INTERVAL_SECONDS: int = 15  # Background pressure check interval

    # Adaptive Batch Size Configuration
    ENABLE_ADAPTIVE_BATCH_SIZE: bool = True  # Enables dynamic batch sizing based on GPU memory availability
    MIN_BATCH_SIZE: int = 1  # Minimum batch size for OOM recovery
    MAX_BATCH_SIZE: int = 256  # Maximum allowed batch size
    BATCH_SIZE_SAFETY_MARGIN: float = 0.2  # 20% safety margin for GPU memory to prevent OOM
    BATCH_SIZE_INCREASE_THRESHOLD: int = 10  # Number of successful batches before attempting to increase size

    # Local LLM Configuration
    ENABLE_LOCAL_LLM: bool = True  # Enable local LLM support in VecPipe
    DEFAULT_LLM_QUANTIZATION: str = "int8"  # Default quantization for local LLMs (int4, int8, float16)
    LLM_UNLOAD_AFTER_SECONDS: int = 300  # Inactivity timeout before LLM is eligible for eviction
    LLM_KV_CACHE_BUFFER_MB: int = 1024  # Conservative KV cache + runtime overhead per loaded LLM
    LLM_TRUST_REMOTE_CODE: bool = False  # Require explicit opt-in for models with remote code

    # Additional Paths specific to vecpipe
    @property
    def operations_dir(self) -> Path:
        return self.data_dir / "operations"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"
