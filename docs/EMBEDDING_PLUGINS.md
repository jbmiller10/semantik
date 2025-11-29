# Embedding Plugins

This project supports a plugin-based architecture for embedding providers, allowing third parties to add new embedding models and providers without modifying the core codebase.

## Overview

The embedding plugin system mirrors the [chunking plugin architecture](./CHUNKING_PLUGINS.md), providing:

- **Auto-detection**: Factory automatically selects the right provider based on model name
- **Built-in providers**: Local dense embeddings (sentence-transformers, Qwen) and mock provider
- **External plugins**: Load custom providers via Python entry points
- **API exposure**: REST endpoints to discover available providers and models

## Architecture

```
packages/shared/embedding/
├── plugin_base.py          # BaseEmbeddingPlugin + EmbeddingProviderDefinition
├── provider_registry.py    # Provider metadata registry
├── factory.py              # EmbeddingProviderFactory with auto-detection
├── plugin_loader.py        # Entry point discovery
├── providers/              # Built-in provider implementations
│   ├── __init__.py         # Auto-registers built-ins on import
│   ├── dense_local.py      # DenseLocalEmbeddingProvider
│   └── mock.py             # MockEmbeddingProvider
├── base.py                 # BaseEmbeddingService (abstract interface)
├── service.py              # Singleton service with factory integration
└── dense.py                # Backward compatibility wrapper
```

## Quick Start

### Using the Factory (Recommended)

```python
from shared.embedding.factory import EmbeddingProviderFactory

# Auto-detect provider based on model name
provider = EmbeddingProviderFactory.create_provider("Qwen/Qwen3-Embedding-0.6B")
await provider.initialize("Qwen/Qwen3-Embedding-0.6B", quantization="float16")

# Generate embeddings
embeddings = await provider.embed_texts(["Hello, world!"])
print(f"Dimension: {provider.get_dimension()}")

# Cleanup
await provider.cleanup()
```

### Using Explicit Provider

```python
from shared.embedding.factory import EmbeddingProviderFactory

# Create specific provider by name
provider = EmbeddingProviderFactory.create_provider_by_name("dense_local")
await provider.initialize("sentence-transformers/all-MiniLM-L6-v2")
```

### Backward Compatibility

Existing code continues to work unchanged:

```python
# Still works
from shared.embedding import EmbeddingService, embedding_service
from shared.embedding.dense import DenseEmbeddingService
```

## Built-in Providers

### DenseLocalEmbeddingProvider

Local embedding generation using sentence-transformers and Qwen models.

| Property | Value |
|----------|-------|
| Internal Name | `dense_local` |
| API ID | `dense_local` |
| Type | `local` |
| Quantization | float32, float16, int8 |
| GPU Support | Yes (CUDA) |

**Supported Model Patterns:**
- `sentence-transformers/*` (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- `BAAI/bge-*` (e.g., `BAAI/bge-large-en-v1.5`)
- `intfloat/*` (e.g., `intfloat/e5-large-v2`)
- `Qwen/Qwen3-Embedding*` (e.g., `Qwen/Qwen3-Embedding-0.6B`)
- Any model in `MODEL_CONFIGS`

**Features:**
- Adaptive batch sizing with OOM recovery
- Instruction-aware embeddings (Qwen models)
- Automatic fallback from GPU to CPU
- Asymmetric embedding support (query vs document mode)

### MockEmbeddingProvider

Deterministic mock embeddings for testing.

| Property | Value |
|----------|-------|
| Internal Name | `mock` |
| API ID | `mock` |
| Type | `local` |
| Quantization | No |
| GPU Support | No (CPU only) |

**Characteristics:**
- Generates deterministic embeddings based on text hash
- Same text always produces same embedding
- Fast, no model downloads required
- Configurable dimension (default: 384)

**Usage:**
```python
provider = EmbeddingProviderFactory.create_provider("mock")
await provider.initialize("mock", dimension=512)
```

## Asymmetric Embedding Mode

Many retrieval models are **asymmetric**, meaning they need different processing for queries (what users search for) vs documents (what gets indexed). The embedding system supports this through the `mode` parameter.

### EmbeddingMode Enum

```python
from shared.embedding.types import EmbeddingMode

# For search queries
embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)

# For document indexing
embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
```

### How It Works

| Model Type | Query Mode | Document Mode |
|------------|-----------|---------------|
| **Qwen** (instruction-based) | Applies `Instruct: {task}\nQuery: {text}` | Raw text (no prefix) |
| **BGE** (prefix-based) | Applies query prefix | No prefix |
| **E5** (prefix-based) | `query: {text}` | `passage: {text}` |
| **Symmetric models** | No transformation | No transformation |

### ModelConfig Asymmetric Fields

When defining model configurations, set these fields for asymmetric models:

```python
ModelConfig(
    name="BAAI/bge-large-en-v1.5",
    dimension=1024,
    is_asymmetric=True,                    # Enable asymmetric handling
    query_prefix="Represent this sentence for searching relevant passages: ",
    document_prefix="",                    # No prefix for documents
    default_query_instruction="",          # For instruction-based models
)
```

### Provider Definition

Providers that support asymmetric mode should declare it:

```python
EmbeddingProviderDefinition(
    ...
    supports_asymmetric=True,  # Provider handles query/document mode
    ...
)
```

### Default Behavior

- If `mode` is not specified, defaults to `QUERY` for backward compatibility
- Document indexing code should explicitly pass `mode=EmbeddingMode.DOCUMENT`
- Search code typically uses `mode=EmbeddingMode.QUERY` (or relies on default)

## Creating External Plugins

### 1. Define the Plugin Class

```python
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from numpy.typing import NDArray
import numpy as np

class MyEmbeddingProvider(BaseEmbeddingPlugin):
    """Custom embedding provider example."""

    # Required class attributes
    INTERNAL_NAME = "my_provider"
    API_ID = "my-provider"
    PROVIDER_TYPE = "local"  # or "remote", "hybrid"

    # Optional metadata for UI/API
    METADATA = {
        "display_name": "My Custom Embeddings",
        "description": "Custom embedding provider using proprietary model",
        "best_for": ["specialized_domain"],
        "pros": ["High accuracy for domain X"],
        "cons": ["Requires specific hardware"],
    }

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.model = None
        self.dimension = 768
        self._initialized = False

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        """Return provider metadata for API exposure."""
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="My Custom Embeddings",
            description="Custom embedding provider",
            provider_type=cls.PROVIDER_TYPE,
            supports_quantization=False,
            supports_instruction=False,
            supports_batch_processing=True,
            supported_models=("my-model-v1", "my-model-v2"),
            default_config={"batch_size": 32},
            is_plugin=True,  # Mark as external plugin
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this provider handles the given model."""
        return model_name.startswith("my-company/") or model_name in ("my-model-v1", "my-model-v2")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, model_name: str, **kwargs) -> None:
        """Initialize the embedding model."""
        # Load your model here
        self.model = self._load_model(model_name)
        self._initialized = True

    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs) -> NDArray[np.float32]:
        """Generate embeddings for texts."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
        # Implement embedding logic
        return np.random.randn(len(texts), self.dimension).astype(np.float32)

    async def embed_single(self, text: str, **kwargs) -> NDArray[np.float32]:
        """Generate embedding for single text."""
        embeddings = await self.embed_texts([text], **kwargs)
        return embeddings[0]

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_info(self) -> dict:
        return {
            "model_name": "my-model",
            "dimension": self.dimension,
            "device": "cpu",
            "provider": self.INTERNAL_NAME,
        }

    async def cleanup(self) -> None:
        """Release resources."""
        self.model = None
        self._initialized = False
```

### 2. Register Entry Point

In your package's `pyproject.toml`:

```toml
[project.entry-points."semantik.embedding_providers"]
my_provider = "my_package.embedding:MyEmbeddingProvider"
```

### 3. Install and Use

```bash
pip install my-embedding-plugin
```

The plugin is automatically discovered and registered at Semantik startup.

## API Endpoints

The embedding plugin system exposes REST endpoints for discovery:

### List Providers

```http
GET /embedding/providers
```

Response:
```json
[
  {
    "id": "dense_local",
    "internal_id": "dense_local",
    "name": "Local Dense Embeddings",
    "description": "Local embedding generation using sentence-transformers or Qwen models",
    "provider_type": "local",
    "supports_quantization": true,
    "supports_instruction": true,
    "supports_batch_processing": true,
    "is_plugin": false
  },
  {
    "id": "mock",
    "internal_id": "mock",
    "name": "Mock Embeddings",
    "description": "Deterministic mock embeddings for testing",
    "provider_type": "local",
    "is_plugin": false
  }
]
```

### Get Provider Details

```http
GET /embedding/providers/{provider_id}
```

### List Models

```http
GET /embedding/models
```

Returns all models from all registered providers.

### Check Model Support

```http
GET /embedding/models/{model_name}/supported
```

Response:
```json
{
  "model_name": "Qwen/Qwen3-Embedding-0.6B",
  "supported": true,
  "provider": "dense_local"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIK_ENABLE_EMBEDDING_PLUGINS` | `true` | Enable/disable plugin loading |

### Disabling Plugins

```bash
export SEMANTIK_ENABLE_EMBEDDING_PLUGINS=false
```

## Plugin Loading

Plugins are loaded at application startup:

1. **WebUI Service**: `packages/webui/startup_tasks.py` calls `ensure_providers_registered()` and `load_embedding_plugins()`
2. **VecPipe Service**: `packages/vecpipe/search/lifespan.py` performs the same registration

### Loading Order

1. Built-in providers registered (import of `providers/__init__.py`)
2. External plugins discovered via entry points
3. Each plugin validated against contract
4. Valid plugins registered with factory and metadata registry

## Plugin Contract

Plugins must provide:

### Required Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `INTERNAL_NAME` | `str` | Internal identifier for factory registration |
| `API_ID` | `str` | API-facing identifier |
| `PROVIDER_TYPE` | `str` | One of: `"local"`, `"remote"`, `"hybrid"` |

### Required Class Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_definition` | `() -> EmbeddingProviderDefinition` | Return provider metadata |
| `supports_model` | `(model_name: str) -> bool` | Check if model is supported |

### Required Instance Methods (from BaseEmbeddingService)

| Method | Description |
|--------|-------------|
| `initialize(model_name, **kwargs)` | Initialize the model |
| `embed_texts(texts, batch_size, **kwargs)` | Embed multiple texts |
| `embed_single(text, **kwargs)` | Embed single text |
| `get_dimension()` | Return embedding dimension |
| `get_model_info()` | Return model metadata dict |
| `cleanup()` | Release resources |
| `is_initialized` (property) | Check initialization status |

## EmbeddingProviderDefinition

The `EmbeddingProviderDefinition` dataclass holds provider metadata:

```python
@dataclass(frozen=True)
class EmbeddingProviderDefinition:
    api_id: str                          # API-facing identifier
    internal_id: str                     # Internal factory name
    display_name: str                    # Human-readable name
    description: str                     # Provider description
    provider_type: str                   # "local", "remote", "hybrid"

    # Capability flags
    supports_quantization: bool = True
    supports_instruction: bool = False
    supports_batch_processing: bool = True
    supports_asymmetric: bool = False    # Query vs document mode handling

    # Model information
    supported_models: tuple[str, ...] = ()

    # Configuration
    default_config: dict[str, Any] = field(default_factory=dict)
    performance_characteristics: dict[str, Any] = field(default_factory=dict)

    # Plugin marker
    is_plugin: bool = False
```

## Troubleshooting

### Plugin Not Loading

1. Check that the entry point is correctly defined in `pyproject.toml`
2. Verify the class is importable: `python -c "from my_package import MyProvider"`
3. Check logs for validation errors during startup
4. Ensure `SEMANTIK_ENABLE_EMBEDDING_PLUGINS` is not set to `false`

### Model Not Detected

1. Verify `supports_model()` returns `True` for your model name
2. Check that the provider is registered:
   ```python
   from shared.embedding.factory import EmbeddingProviderFactory
   print(EmbeddingProviderFactory.list_available_providers())
   ```

### Provider Conflicts

If multiple providers claim to support the same model, the first registered provider wins. Built-in providers are registered before external plugins.

## Security

- Plugins run in-process and are trusted code
- Do not install untrusted plugins without thorough review
- Plugins have full access to the Python runtime
- Validate any external model downloads in your plugin

## Related Documentation

- [Chunking Plugins](./CHUNKING_PLUGINS.md) - Similar plugin architecture for chunking
- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Architecture](./ARCH.md) - System architecture overview
