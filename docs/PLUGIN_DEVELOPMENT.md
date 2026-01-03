# Semantik Plugin Development Guide

This guide explains how to develop plugins for Semantik's self-hosted semantic search engine.

## Overview

Semantik plugins extend core functionality through a unified plugin system. All plugins inherit from `SemanticPlugin` and can be distributed via pip packages.

### Plugin Types

| Type | Purpose | Base Class |
|------|---------|------------|
| **embedding** | Custom embedding models | `BaseEmbeddingPlugin` |
| **chunking** | Document chunking strategies | `ChunkingPlugin` |
| **connector** | Data source integrations | `ConnectorPlugin` |
| **reranker** | Search result reranking | `RerankerPlugin` |
| **extractor** | Metadata extraction | `ExtractorPlugin` |

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
