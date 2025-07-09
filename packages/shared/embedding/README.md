# Embedding Service Module

This module provides a flexible, extensible embedding service architecture for the Semantik project.

## Architecture Overview

```
embedding/
├── base.py          # Abstract base class defining the interface
├── dense.py         # Dense embedding implementation (sentence-transformers, Qwen)
├── service.py       # Singleton service management
├── models.py        # Centralized model configurations
└── __init__.py      # Public API exports
```

## Usage Examples

### Basic Usage (Synchronous)

```python
from shared.embedding import embedding_service

# Load a model
embedding_service.load_model("BAAI/bge-large-en-v1.5", quantization="float16")

# Generate embeddings
texts = ["Hello world", "How are you?"]
embeddings = embedding_service.generate_embeddings(
    texts, 
    model_name="BAAI/bge-large-en-v1.5",
    batch_size=32
)
```

### Async Usage

```python
from shared.embedding import get_embedding_service, initialize_embedding_service

# Initialize with specific model
service = await initialize_embedding_service(
    "Qwen/Qwen3-Embedding-0.6B",
    quantization="float16"
)

# Generate embeddings
embeddings = await service.embed_texts(
    ["Query text"], 
    instruction="Represent this text for similarity search"
)
```

### Implementing a New Embedding Service

To add a new embedding service (e.g., sparse embeddings), create a new class inheriting from `BaseEmbeddingService`:

```python
from shared.embedding.base import BaseEmbeddingService
import numpy as np

class SparseEmbeddingService(BaseEmbeddingService):
    """Example sparse embedding service implementation."""
    
    def __init__(self):
        self._initialized = False
        self.vocab_size = None
        self.model = None
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    async def initialize(self, model_name: str, **kwargs):
        """Initialize the sparse embedding model."""
        # Load your sparse model here
        self.vocab_size = kwargs.get("vocab_size", 50000)
        self.model = load_sparse_model(model_name)
        self._initialized = True
    
    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Generate sparse embeddings."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        # Your sparse embedding logic here
        sparse_vectors = self.model.encode(texts)
        return sparse_vectors
    
    async def embed_single(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding for single text."""
        return (await self.embed_texts([text], **kwargs))[0]
    
    def get_dimension(self) -> int:
        """Return vocabulary size for sparse embeddings."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return self.vocab_size
    
    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_name": self.model_name,
            "dimension": self.vocab_size,
            "type": "sparse",
            "device": "cpu"
        }
    
    async def cleanup(self):
        """Clean up resources."""
        self.model = None
        self._initialized = False
```

### Adding New Models

To add support for new models, update `models.py`:

```python
from shared.embedding.models import ModelConfig, add_model_config

# Add a new model configuration
add_model_config(ModelConfig(
    name="your-org/your-model",
    dimension=768,
    description="Description of your model",
    max_sequence_length=512,
    supports_quantization=True,
    recommended_quantization="float16",
    memory_estimate={"float32": 400, "float16": 200, "int8": 100}
))
```

## Design Patterns

### Singleton Pattern

The `service.py` module implements a singleton pattern to ensure only one embedding service instance exists:

```python
# Global instance management
_embedding_service: BaseEmbeddingService | None = None
_service_lock = asyncio.Lock()

async def get_embedding_service() -> BaseEmbeddingService:
    """Get or create the singleton embedding service."""
    global _embedding_service
    
    async with _service_lock:
        if _embedding_service is None:
            _embedding_service = DenseEmbeddingService()
    
    return _embedding_service
```

This ensures:
- Memory efficiency (only one model loaded)
- Thread-safe access
- Consistent state across the application

### Async/Sync Bridge

The `EmbeddingService` class in `dense.py` provides a synchronous wrapper around the async implementation:

```python
def load_model(self, model_name: str, quantization: str = "float32") -> bool:
    """Load a model synchronously."""
    loop = self._get_loop()
    loop.run_until_complete(
        self._service.initialize(model_name, quantization=quantization)
    )
```

This allows gradual migration from sync to async code.

## Configuration

Model configurations are centralized in `models.py`. Each model has:
- Dimension
- Max sequence length  
- Quantization support
- Memory estimates
- Model-specific settings

## Error Handling

The service includes comprehensive error handling:

1. **Initialization errors**: Clear messages if model loading fails
2. **Quantization fallback**: Automatic fallback from INT8 to float32 if needed
3. **OOM recovery**: Adaptive batch sizing (in the old implementation, could be added)
4. **Runtime checks**: Ensures service is initialized before use

## Performance Considerations

1. **Model Loading**: Models are loaded on-demand and cached
2. **Batch Processing**: Configurable batch sizes for optimal throughput
3. **Memory Management**: Explicit cleanup methods to free GPU memory
4. **Quantization**: Support for float16 and int8 to reduce memory usage

## Testing

Run tests with:
```bash
pytest tests/test_embedding_*.py -v
```

Tests cover:
- Basic functionality
- Async/sync interaction
- Concurrent requests
- Configuration management
- Backwards compatibility