# Embedding Plugins

Add custom embedding providers without touching core code.

## Overview

- Auto-detect provider by model name
- Built-in: sentence-transformers, Qwen, mock
- External: Load via Python entry points
- REST API for discovery

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

### dense_local
sentence-transformers + Qwen. Supports quantization (float32/16, int8), GPU, adaptive batching, instruction-aware (Qwen), asymmetric mode.

**Models**: `sentence-transformers/*`, `BAAI/bge-*`, `intfloat/*`, `Qwen/Qwen3-Embedding*`

### mock
Deterministic test embeddings (hash-based, no downloads, CPU-only, configurable dimension).

## Asymmetric Mode

Some models need different processing for queries vs documents.

```python
from shared.embedding.types import EmbeddingMode

# Query (default)
await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)

# Documents
await provider.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
```

**Examples**:
- Qwen: adds instruction prefix to queries, raw for docs
- BGE/E5: different prefixes
- Symmetric: no transformation

Set `is_asymmetric=True` and prefixes in `ModelConfig` for custom models.

## Creating Plugins

1. **Subclass** `BaseEmbeddingPlugin`
2. **Implement** required methods: `initialize`, `embed_texts`, `embed_single`, `get_dimension`, `get_model_info`, `cleanup`
3. **Define** `get_definition()` classmethod returning `EmbeddingProviderDefinition`
4. **Register** entry point in `pyproject.toml`:

```toml
[project.entry-points."semantik.embedding_providers"]
my_provider = "my_package.embedding:MyProvider"
```

See EMBEDDING_PLUGINS.md source for full example (removed to save space).

## API Endpoints

**GET /embedding/providers** - List all providers
**GET /embedding/providers/{id}** - Provider details
**GET /embedding/models** - All models
**GET /embedding/models/{name}/supported** - Check support
**GET /api/models** - UI model discovery (includes plugins)

## Configuration

Toggle via `SEMANTIK_ENABLE_EMBEDDING_PLUGINS` (default `true`).

Loaded at startup in webui and vecpipe services. Built-ins first, then external plugins via entry points.

## Plugin Contract

**Class attrs**: `INTERNAL_NAME`, `API_ID`, `PROVIDER_TYPE`
**Class methods**: `get_definition()`, `supports_model()`
**Instance methods**: `initialize`, `embed_texts`, `embed_single`, `get_dimension`, `get_model_info`, `cleanup`, `is_initialized`

## Troubleshooting

**Not loading?** Check entry point, importability, logs, env var.
**Model not detected?** Verify `supports_model()` returns `True`.
**Conflicts?** First registered wins (built-ins before plugins).

## Security

Plugins run in-process. Review untrusted code.
