# Semantik Plugin Development Guide

This guide explains how to develop plugins for Semantik's self-hosted semantic search engine.

## Overview

Semantik plugins extend core functionality through a unified plugin system. Plugins can be distributed via pip packages and support two development approaches:

### Development Approaches

| Approach | Best For | Documentation |
|----------|----------|---------------|
| **Protocol-based** (Recommended) | External plugins without semantik dependencies | [External Plugins Guide](external-plugins.md) |
| **ABC-based** (This guide) | Internal plugins or when you need semantik utilities | This document |

**New in v0.8**: External plugins can now be developed with **zero semantik imports** using Python's structural typing (Protocols). See [External Plugins Guide](external-plugins.md) for the protocol-based approach.

This guide covers the ABC-based approach where plugins inherit from `SemanticPlugin` base classes.

### Plugin Types

| Type | Purpose | Base Class |
|------|---------|------------|
| **embedding** | Custom embedding models | `BaseEmbeddingPlugin` |
| **chunking** | Document chunking strategies | `ChunkingPlugin` |
| **connector** | Data source integrations | `ConnectorPlugin` |
| **reranker** | Search result reranking | `RerankerPlugin` |
| **extractor** | Metadata extraction | `ExtractorPlugin` |
| **sparse_indexer** | Sparse vectors (BM25/SPLADE) | `SparseIndexerPlugin` |

## Quick Start

### 1. Create Project Structure

```
semantik-plugin-myembedding/
├── pyproject.toml
├── my_plugin/
│   ├── __init__.py
│   └── provider.py
└── tests/
    └── test_provider.py
```

### 2. Define Entry Point

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "semantik-plugin-myembedding"
version = "1.0.0"
dependencies = [
    "semantik-shared>=0.7.0",  # or copy needed base classes
]

[project.entry-points."semantik.plugins"]
my-embedding = "my_plugin.provider:MyEmbeddingProvider"
```

### 3. Implement Plugin Class

```python
# my_plugin/provider.py
from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest

class MyPlugin(SemanticPlugin):
    PLUGIN_TYPE = "embedding"  # or chunking, connector, etc.
    PLUGIN_ID = "my-plugin"
    PLUGIN_VERSION = "1.0.0"

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name="My Plugin",
            description="A custom plugin for Semantik",
        )
```

### 4. Install and Test

```bash
# Install in development mode
pip install -e ./semantik-plugin-myembedding

# Or install in Semantik via the UI/API
# POST /api/v2/plugins/install
# {"install_command": "git+https://github.com/user/semantik-plugin-myembedding.git"}
```

---

## Plugin Base Class

All plugins must inherit from `SemanticPlugin`:

```python
from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest

class SemanticPlugin(ABC):
    """Universal base for all Semantik plugins."""

    # Required class attributes
    PLUGIN_TYPE: ClassVar[str]      # "embedding", "chunking", etc.
    PLUGIN_ID: ClassVar[str]        # Unique identifier (lowercase, hyphens)
    PLUGIN_VERSION: ClassVar[str]   # Semantic version (default: "0.0.0")

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with optional configuration."""

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin metadata for discovery."""

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        """Return JSON Schema for configuration (optional)."""

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
        """Return True if plugin is operational (optional)."""

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize plugin resources (optional)."""

    async def cleanup(self) -> None:
        """Clean up plugin resources (optional)."""

    @property
    def is_initialized(self) -> bool:
        """Check if plugin has been initialized."""
```

---

## Embedding Plugins

Embedding plugins provide custom text embedding capabilities.

```python
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.embedding.types import EmbeddingMode

class MyEmbeddingProvider(BaseEmbeddingPlugin):
    INTERNAL_NAME = "my_embeddings"
    API_ID = "my-embeddings"
    PROVIDER_TYPE = "remote"  # "local", "remote", or "hybrid"
    PLUGIN_VERSION = "1.0.0"
    METADATA = {
        "display_name": "My Embeddings",
        "description": "Custom embedding provider using My API",
    }

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            provider_type=cls.PROVIDER_TYPE,
            supports_quantization=False,
            supports_asymmetric=True,  # Different processing for queries vs docs
            is_plugin=True,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this provider supports the given model."""
        return model_name.startswith("my-model-")

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "api_key_env": {
                    "type": "string",
                    "description": "Environment variable containing API key",
                },
                "model": {
                    "type": "string",
                    "default": "my-model-small",
                    "enum": ["my-model-small", "my-model-large"],
                },
            },
            "required": ["api_key_env"],
        }

    async def embed_texts(
        self,
        texts: list[str],
        mode: EmbeddingMode = EmbeddingMode.QUERY,
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        # Implement your embedding logic here
        pass

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return 768
```

---

## Connector Plugins

Connector plugins integrate external data sources.

```python
from shared.plugins.types.connector import ConnectorPlugin

class MyConnector(ConnectorPlugin):
    PLUGIN_TYPE = "connector"
    PLUGIN_ID = "my-connector"
    PLUGIN_VERSION = "1.0.0"
    METADATA = {
        "name": "My Data Source",
        "description": "Connect to My Data Source",
        "icon": "database",
        "supports_sync": True,
    }

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Return configuration field definitions for the UI."""
        return [
            {
                "name": "base_url",
                "type": "text",
                "label": "Base URL",
                "description": "The base URL of the data source",
                "required": True,
                "placeholder": "https://api.example.com",
            },
            {
                "name": "folder_id",
                "type": "text",
                "label": "Folder ID",
                "description": "Optional folder to limit sync scope",
                "required": False,
            },
        ]

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Return secret field definitions for the UI."""
        return [
            {
                "name": "api_key",
                "label": "API Key",
                "description": "API key for authentication",
                "required": True,
            },
        ]

    async def authenticate(self) -> bool:
        """Verify credentials are valid."""
        # Implement authentication check
        return True

    async def load_documents(self) -> AsyncIterator[Document]:
        """Yield documents from the data source."""
        # Implement document loading
        pass
```

### Parsing Binary Files in Connectors

If your connector handles binary file formats (PDF, DOCX, etc.), use the parser system to extract text:

```python
from pathlib import Path
from shared.text_processing.parsers import (
    parse_content,
    ExtractionFailedError,
    UnsupportedFormatError,
)

async def load_documents(self, source_id=None) -> AsyncIterator[IngestedDocument]:
    for file_bytes, filename in self._fetch_files():
        try:
            result = parse_content(
                file_bytes,
                filename=filename,
                file_extension=Path(filename).suffix,
                metadata={"source_type": self.PLUGIN_ID, "source_path": filename},
            )
            yield IngestedDocument(
                content=result.text,
                unique_id=filename,
                source_type=self.PLUGIN_ID,
                metadata=result.metadata,
                content_hash=hashlib.sha256(file_bytes).hexdigest(),
            )
        except UnsupportedFormatError:
            continue  # Skip unsupported files
        except ExtractionFailedError as e:
            logger.error(f"Failed to parse {filename}: {e}")
```

See [PARSERS.md](./PARSERS.md) for parser selection rules and configuration options.

---

## Reranker Plugins

Reranker plugins reorder search results based on relevance.

```python
from shared.plugins.types.reranker import RerankerPlugin, RerankResult, RerankerCapabilities

class MyReranker(RerankerPlugin):
    PLUGIN_TYPE = "reranker"
    PLUGIN_ID = "my-reranker"
    PLUGIN_VERSION = "1.0.0"
    METADATA = {
        "display_name": "My Reranker",
        "description": "Cross-encoder based reranking",
    }

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            capabilities=cls.get_capabilities().__dict__,
        )

    @classmethod
    def get_capabilities(cls) -> RerankerCapabilities:
        return RerankerCapabilities(
            max_documents=100,
            max_query_length=512,
            supports_batching=True,
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query."""
        # Implement reranking logic
        pass
```

---

## Extractor Plugins

> **Note**: Extractors operate on *already-parsed text* to extract metadata like entities, keywords, and sentiment. To convert documents (PDF, DOCX, etc.) to text, use the parser system - see [PARSERS.md](./PARSERS.md). Parsers are built-in utilities, not plugins.

Extractor plugins extract metadata from documents.

```python
from shared.plugins.types.extractor import (
    ExtractorPlugin,
    ExtractionType,
    ExtractionResult,
    Entity,
)

class MyExtractor(ExtractorPlugin):
    PLUGIN_TYPE = "extractor"
    PLUGIN_ID = "my-extractor"
    PLUGIN_VERSION = "1.0.0"
    METADATA = {
        "display_name": "My Extractor",
        "description": "Extract custom entities from text",
    }

    @classmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        return [ExtractionType.ENTITIES, ExtractionType.KEYWORDS]

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and keywords from text."""
        return ExtractionResult(
            entities=[
                Entity(text="Example", label="ORG", start=0, end=7),
            ],
            keywords=["example", "keyword"],
        )
```

---

## Sparse Indexer Plugins

Sparse indexer plugins generate sparse vector representations for hybrid search. Two main types are supported:

- **BM25**: Classic term-frequency based retrieval (stateful, requires IDF statistics)
- **SPLADE**: Learned sparse representations using neural models (stateless)

```python
from shared.plugins.types.sparse_indexer import (
    SparseIndexerPlugin,
    SparseIndexerCapabilities,
    SparseVector,
    SparseQueryVector,
)

class MyBM25Indexer(SparseIndexerPlugin):
    PLUGIN_TYPE = "sparse_indexer"
    PLUGIN_ID = "my-bm25"
    PLUGIN_VERSION = "1.0.0"
    SPARSE_TYPE = "bm25"  # Must be "bm25" or "splade"
    METADATA = {
        "display_name": "My BM25 Indexer",
        "description": "Custom BM25 implementation for sparse retrieval",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._k1 = (config or {}).get("k1", 1.5)
        self._b = (config or {}).get("b", 0.75)
        self._vocabulary: dict[str, int] = {}
        self._idf_stats: dict[str, float] = {}

    async def encode_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[SparseVector]:
        """Generate sparse vectors for documents.

        Args:
            documents: List of dicts with 'content' and 'chunk_id' keys.

        Returns:
            List of SparseVector instances (one per document).
        """
        results = []
        for doc in documents:
            # Tokenize and compute BM25 scores
            tokens = self._tokenize(doc["content"])
            indices, values = self._compute_bm25_vector(tokens)
            results.append(SparseVector(
                indices=tuple(indices),
                values=tuple(values),
                chunk_id=doc["chunk_id"],
            ))
        return results

    async def encode_query(self, query: str) -> SparseQueryVector:
        """Generate sparse vector for a search query."""
        tokens = self._tokenize(query)
        indices, values = self._compute_query_vector(tokens)
        return SparseQueryVector(
            indices=tuple(indices),
            values=tuple(values),
        )

    async def remove_documents(self, chunk_ids: list[str]) -> None:
        """Update IDF statistics when chunks are removed.

        For BM25, this updates corpus statistics. For SPLADE, this is a no-op.
        """
        await self._update_idf_for_removal(chunk_ids)

    @classmethod
    def get_capabilities(cls) -> SparseIndexerCapabilities:
        """Declare indexer capabilities and limits."""
        return SparseIndexerCapabilities(
            sparse_type="bm25",
            max_tokens=8192,
            vocabulary_handling="direct",  # or "hashed" for large vocabularies
            supports_batching=True,
            max_batch_size=64,
            requires_corpus_stats=True,  # BM25 needs IDF
            idf_storage="file",  # or "qdrant_point"
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms."""
        # Implement tokenization (lowercase, remove stopwords, etc.)
        pass

    def _compute_bm25_vector(self, tokens: list[str]) -> tuple[list[int], list[float]]:
        """Compute BM25 sparse vector from tokens."""
        # Implement BM25 scoring
        pass

    def _compute_query_vector(self, tokens: list[str]) -> tuple[list[int], list[float]]:
        """Compute query vector (typically just IDF weights)."""
        pass

    async def _update_idf_for_removal(self, chunk_ids: list[str]) -> None:
        """Update IDF statistics when chunks are removed."""
        pass
```

### Key Concepts

**SPARSE_TYPE**: Must be either `"bm25"` or `"splade"`. This determines the naming convention for sparse Qdrant collections (e.g., `collection_sparse_bm25`).

**Constraint**: Only one sparse indexer per collection. Users who need both BM25 and SPLADE should create separate collections.

**Stateful vs Stateless**:
- **BM25**: Stateful - requires corpus IDF statistics stored in files or Qdrant
- **SPLADE**: Stateless - model parameters encode all knowledge, no corpus stats needed

### Configuration Schema

```python
@classmethod
def get_config_schema(cls) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "k1": {
                "type": "number",
                "description": "Term saturation parameter (higher = more TF weight)",
                "default": 1.5,
                "minimum": 0,
                "maximum": 3,
            },
            "b": {
                "type": "number",
                "description": "Length normalization (0 = none, 1 = full)",
                "default": 0.75,
                "minimum": 0,
                "maximum": 1,
            },
            "collection_name": {
                "type": "string",
                "description": "Collection name for IDF persistence",
            },
        },
    }
```

### Built-in Implementations

Semantik includes two built-in sparse indexers:

| Plugin ID | Type | Description |
|-----------|------|-------------|
| `bm25-local` | BM25 | Classic BM25 with configurable k1/b, stopword removal |
| `splade-local` | SPLADE | Neural sparse encoder using `naver/splade-cocondenser-ensembledistil` |

See [Sparse Indexing Guide](SPARSE_INDEXING.md) for usage details and performance considerations.

---

## Configuration

### JSON Schema

Define configuration with JSON Schema for validation and UI generation:

```python
@classmethod
def get_config_schema(cls) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Model name to use",
                "default": "default-model",
            },
            "batch_size": {
                "type": "integer",
                "description": "Batch size for processing",
                "minimum": 1,
                "maximum": 100,
                "default": 32,
            },
        },
        "required": ["model"],
    }
```

### Secrets Handling

Never store API keys directly. Use the `_env` suffix pattern:

```python
# Configuration stored in database:
{
    "api_key_env": "MY_API_KEY",  # Reference to environment variable
    "model": "my-model-v1"
}

# At runtime, Semantik resolves the env var:
resolved_config = {
    "api_key": "actual-api-key-value",  # Resolved from MY_API_KEY env var
    "model": "my-model-v1"
}
```

In your plugin, always get secrets from the resolved config:

```python
async def initialize(self, config: dict[str, Any] | None = None) -> None:
    await super().initialize(config)
    self.api_key = self._config.get("api_key")  # Already resolved
```

---

## Health Checks

Implement health checks to verify plugin functionality:

```python
@classmethod
async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
    """Return True if the plugin is operational.

    Health checks should:
    - Complete within 5 seconds (timeout enforced)
    - Not modify any state
    - Verify external dependencies are accessible
    """
    if not config:
        return False

    api_key = config.get("api_key")
    if not api_key:
        return False

    try:
        # Verify API connectivity
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.example.com/health",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=3),
            ) as response:
                return response.status == 200
    except Exception:
        return False
```

---

## Dependencies

Declare dependencies on other plugins:

```python
@classmethod
def get_manifest(cls) -> PluginManifest:
    return PluginManifest(
        id=cls.PLUGIN_ID,
        type=cls.PLUGIN_TYPE,
        version=cls.PLUGIN_VERSION,
        display_name="My Plugin",
        description="Depends on another plugin",
        requires=[
            "other-plugin",  # Simple dependency
            {
                "plugin_id": "versioned-plugin",
                "min_version": "1.0.0",
                "max_version": "2.0.0",
            },
            {
                "plugin_id": "optional-plugin",
                "optional": True,
            },
        ],
    )
```

Dependencies are validated at load time (warnings logged, not blocking).

---

## Testing

```python
import pytest
from my_plugin.provider import MyPlugin

class TestMyPlugin:
    def test_manifest(self):
        manifest = MyPlugin.get_manifest()
        assert manifest.id == "my-plugin"
        assert manifest.type == "embedding"
        assert manifest.version == "1.0.0"

    def test_config_schema(self):
        schema = MyPlugin.get_config_schema()
        assert schema is not None
        assert "properties" in schema
        assert "api_key_env" in schema["properties"]

    @pytest.mark.asyncio
    async def test_health_check_without_config(self):
        result = await MyPlugin.health_check(None)
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_config(self, mock_api):
        result = await MyPlugin.health_check({
            "api_key": "test-key",
        })
        assert result is True

    @pytest.mark.asyncio
    async def test_embed_texts(self):
        plugin = MyPlugin({"api_key": "test"})
        await plugin.initialize()

        embeddings = await plugin.embed_texts(["Hello world"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == plugin.get_dimension()
```

---

## Distribution

### Via Git URL

Users can install directly from a git repository:

```bash
# Via Semantik API
POST /api/v2/plugins/install
{"install_command": "git+https://github.com/user/semantik-plugin-myembedding.git"}

# With specific version
{"install_command": "git+https://github.com/user/semantik-plugin-myembedding.git@v1.0.0"}
```

### Via PyPI

Publish to PyPI for easier distribution:

```bash
pip install semantik-plugin-myembedding
```

---

## Troubleshooting

### Plugin Not Loading

1. Verify entry point is registered:
   ```bash
   pip show semantik-plugin-myembedding
   # Check entry points section
   ```

2. Check environment variable:
   ```bash
   echo $SEMANTIK_ENABLE_PLUGINS  # Should be "true" or unset
   ```

3. Check logs:
   ```bash
   docker logs semantik-webui 2>&1 | grep -i plugin
   ```

### Health Check Failing

1. Verify configuration in the UI
2. Check environment variables are set:
   ```bash
   docker exec semantik-webui env | grep API_KEY
   ```
3. Test API connectivity manually

### Import Errors

Ensure your plugin doesn't import Semantik internals that may not be available:

```python
# Good - conditional import
try:
    from shared.plugins.base import SemanticPlugin
except ImportError:
    # Provide fallback or raise helpful error
    raise ImportError("semantik-shared package required")
```

---

## Best Practices

1. **Keep health checks fast** - Complete within 3 seconds
2. **Handle missing config gracefully** - Return False from health_check if config is missing
3. **Use semantic versioning** - Follow semver for PLUGIN_VERSION
4. **Document configuration** - Provide clear descriptions in config schema
5. **Test thoroughly** - Include unit tests for all public methods
6. **Log appropriately** - Use Python logging, not print statements
7. **Clean up resources** - Implement cleanup() to release connections/memory

---

## See Also

- [External Plugins Guide](external-plugins.md) - Protocol-based development (no semantik imports)
- [Plugin Protocols Reference](plugin-protocols.md) - Complete protocol specifications
- [Plugin Testing](PLUGIN_TESTING.md) - Testing infrastructure and contract tests
- [Plugin Security](PLUGIN_SECURITY.md) - Security considerations for plugins
