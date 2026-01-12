# Creating External Plugins

External plugins can be developed without any semantik dependencies. This guide shows how to create plugins using only Python standard library types.

## Overview

Semantik uses **structural typing** (duck typing via Python Protocols) for plugin validation. Your plugin class just needs to have the right methods and attributes - no imports from semantik required.

This means you can:
- Develop plugins in a separate repository
- Use any Python version 3.10+
- Avoid dependency conflicts with semantik internals
- Distribute plugins via PyPI or git repositories

## Protocol Version

Current protocol version: **1.0.0**

Breaking changes to protocols will increment the major version. Your plugins will continue to work as long as they satisfy the protocol interface.

---

## Quick Start: Connector Plugin

Here's a minimal working connector plugin with zero semantik imports:

```python
# my_connector/plugin.py
"""A document source connector - no semantik imports required."""
from typing import ClassVar, Any, AsyncIterator
import hashlib


class MyConnector:
    """Document source connector using structural typing."""

    # Required class variables
    PLUGIN_ID: ClassVar[str] = "my-connector"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    METADATA: ClassVar[dict[str, Any]] = {
        "name": "My Connector",
        "description": "Connects to my data source",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    async def authenticate(self) -> bool:
        """Verify credentials are valid."""
        return self._config.get("api_key") is not None

    async def load_documents(
        self, source_id: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield documents from the source."""
        content = "Document text content from my source..."
        yield {
            "content": content,
            "unique_id": "doc-123",
            "source_type": "my-connector",
            "metadata": {"source": "example"},
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Define configuration fields for the UI."""
        return [
            {"name": "api_key", "type": "password", "label": "API Key", "required": True},
        ]

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Define which fields contain secrets."""
        return [{"name": "api_key"}]

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery."""
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "My Connector",
            "description": "Connects to my data source",
        }
```

```toml
# pyproject.toml
[project]
name = "my-connector"
version = "1.0.0"
requires-python = ">=3.10"

[project.entry-points."semantik.plugins"]
my-connector = "my_connector.plugin:MyConnector"
```

Install and test:

```bash
pip install -e ./my-connector
python -c "
from my_connector.plugin import MyConnector
print(f'Loaded: {MyConnector.PLUGIN_ID}')
print(f'Manifest: {MyConnector.get_manifest()}')
"
```

---

## Plugin Types

Semantik supports 6 plugin types:

| Type | Purpose | Key Method |
|------|---------|------------|
| `connector` | Document source ingestion | `load_documents()` |
| `embedding` | Text to vector conversion | `embed_texts()` |
| `chunking` | Text segmentation | `chunk()` |
| `reranker` | Search result reordering | `rerank()` |
| `extractor` | Entity/metadata extraction | `extract()` |

---

## Complete Examples

### Connector Plugin

Connectors ingest documents from external sources.

```python
from typing import ClassVar, Any, AsyncIterator
import hashlib


class MyConnector:
    PLUGIN_ID: ClassVar[str] = "my-connector"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    METADATA: ClassVar[dict[str, Any]] = {
        "name": "My Connector",
        "description": "Connects to my data source",
        "icon": "database",
        "supports_sync": True,
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    async def authenticate(self) -> bool:
        """Verify credentials are valid."""
        api_key = self._config.get("api_key")
        if not api_key:
            return False
        # Validate with your API...
        return True

    async def load_documents(
        self, source_id: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield documents from the source.

        Each document must have:
        - content: Full text content
        - unique_id: Unique identifier
        - source_type: Your PLUGIN_ID
        - metadata: Source-specific metadata dict
        - content_hash: SHA-256 hash (64 lowercase hex chars)
        - file_path: (optional) Local file path
        """
        # Fetch documents from your source...
        documents = [
            {"id": "doc-1", "text": "First document content..."},
            {"id": "doc-2", "text": "Second document content..."},
        ]

        for doc in documents:
            content = doc["text"]
            yield {
                "content": content,
                "unique_id": doc["id"],
                "source_type": self.PLUGIN_ID,
                "metadata": {"original_id": doc["id"]},
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            }

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Define configuration fields for the UI."""
        return [
            {
                "name": "base_url",
                "type": "text",
                "label": "Base URL",
                "description": "API endpoint URL",
                "required": True,
                "placeholder": "https://api.example.com",
            },
            {
                "name": "folder_id",
                "type": "text",
                "label": "Folder ID",
                "description": "Optional folder to limit scope",
                "required": False,
            },
        ]

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Define which fields contain secrets (encrypted in storage)."""
        return [
            {
                "name": "api_key",
                "label": "API Key",
                "description": "API key for authentication",
                "required": True,
            },
        ]

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": cls.METADATA["name"],
            "description": cls.METADATA["description"],
            "author": "Your Name",
            "homepage": "https://github.com/you/my-connector",
        }
```

### Embedding Plugin

Embedding plugins convert text to vector representations.

```python
from typing import ClassVar, Any


class MyEmbedding:
    PLUGIN_ID: ClassVar[str] = "my-embedding"
    PLUGIN_TYPE: ClassVar[str] = "embedding"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    INTERNAL_NAME: ClassVar[str] = "my_embedding"
    API_ID: ClassVar[str] = "my-embedding"
    PROVIDER_TYPE: ClassVar[str] = "remote"  # "local", "remote", or "hybrid"
    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "My Embedding",
        "description": "Custom embedding provider",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._dimension = 384

    async def embed_texts(
        self,
        texts: list[str],
        mode: str = "document",
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed.
            mode: "query" or "document".
                  Query mode may apply prefixes for asymmetric models.

        Returns:
            List of embedding vectors (one per input text).
        """
        # Call your embedding API/model...
        # This example returns fixed-size mock vectors
        return [[0.1] * self._dimension for _ in texts]

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        """Return provider definition for registration."""
        return {
            "api_id": cls.API_ID,
            "internal_id": cls.INTERNAL_NAME,
            "display_name": cls.METADATA["display_name"],
            "description": cls.METADATA["description"],
            "provider_type": cls.PROVIDER_TYPE,
            "supports_asymmetric": True,
            "is_plugin": True,
        }

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this provider supports a specific model."""
        return model_name.startswith("my-model-")

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": cls.METADATA["display_name"],
            "description": cls.METADATA["description"],
        }
```

### Chunking Plugin

Chunking plugins split documents into smaller segments.

```python
from typing import ClassVar, Any, Callable


class MyChunking:
    PLUGIN_ID: ClassVar[str] = "my-chunking"
    PLUGIN_TYPE: ClassVar[str] = "chunking"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    def chunk(
        self,
        content: str,
        config: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Split content into chunks.

        Args:
            content: Full text to chunk.
            config: Chunking config with chunk_size, chunk_overlap, etc.
            progress_callback: Optional callback for progress (0.0-1.0).

        Returns:
            List of chunk dicts with content and metadata.
        """
        chunk_size = config.get("chunk_size", 500)
        overlap = config.get("chunk_overlap", 50)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "chunk_index": chunk_index,
                    "start_offset": start,
                    "end_offset": end,
                },
            })

            if progress_callback:
                progress_callback(min(end / len(content), 1.0))

            start = end - overlap
            chunk_index += 1

        return chunks

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate content before chunking.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        if not content or not content.strip():
            return False, "Content cannot be empty"
        return True, None

    def estimate_chunks(self, content_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for given content length."""
        chunk_size = config.get("chunk_size", 500)
        overlap = config.get("chunk_overlap", 50)
        if chunk_size <= overlap:
            return 1
        return max(1, (content_length + chunk_size - overlap - 1) // (chunk_size - overlap))

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "My Chunking",
            "description": "Custom chunking strategy",
        }
```

### Reranker Plugin

Reranker plugins reorder search results by relevance.

```python
from typing import ClassVar, Any


class MyReranker:
    PLUGIN_ID: ClassVar[str] = "my-reranker"
    PLUGIN_TYPE: ClassVar[str] = "reranker"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents for a query.

        Args:
            query: Search query string.
            documents: List of document texts to rerank.
            top_k: Optional limit on results.
            metadata: Optional metadata for each document.

        Returns:
            List of result dicts with index and score.
        """
        # Your reranking logic here...
        # This example returns documents in original order with mock scores
        results = []
        for i, doc in enumerate(documents):
            score = 1.0 - (i * 0.1)  # Decreasing scores
            results.append({
                "index": i,
                "score": score,
                "text": doc,
            })

        if top_k:
            results = results[:top_k]

        return results

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare reranker capabilities."""
        return {
            "max_documents": 100,
            "max_query_length": 512,
            "max_doc_length": 4096,
            "supports_batching": True,
        }

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "My Reranker",
            "description": "Custom reranker plugin",
        }
```

### Extractor Plugin

Extractor plugins extract structured information from text.

```python
from typing import ClassVar, Any


class MyExtractor:
    PLUGIN_ID: ClassVar[str] = "my-extractor"
    PLUGIN_TYPE: ClassVar[str] = "extractor"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    async def extract(
        self,
        text: str,
        extraction_types: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract information from text.

        Args:
            text: Input text to extract from.
            extraction_types: Types to extract (see valid values below).
            options: Additional extraction options.

        Returns:
            Dictionary with extracted data.
        """
        types = extraction_types or ["keywords", "entities"]
        result: dict[str, Any] = {}

        if "keywords" in types:
            # Your keyword extraction logic...
            result["keywords"] = ["example", "keyword"]

        if "entities" in types:
            # Your entity extraction logic...
            result["entities"] = [
                {"text": "Example Corp", "type": "ORG", "confidence": 0.95},
            ]

        if "language" in types:
            result["language"] = "en"
            result["language_confidence"] = 0.99

        if "sentiment" in types:
            result["sentiment"] = 0.5  # -1.0 to 1.0

        if "summary" in types:
            result["summary"] = text[:100] + "..."

        return result

    @classmethod
    def supported_extractions(cls) -> list[str]:
        """List supported extraction types."""
        return ["keywords", "entities", "language", "sentiment", "summary"]

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "My Extractor",
            "description": "Custom extractor plugin",
        }
```

---

## Data Format Specifications

### Document Format (Connectors)

Connectors must yield documents matching `IngestedDocumentDict`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | `str` | Yes | Full text content of the document |
| `unique_id` | `str` | Yes | Unique identifier (URI, path, message ID) |
| `source_type` | `str` | Yes | Your connector's PLUGIN_ID |
| `metadata` | `dict[str, Any]` | Yes | Source-specific metadata |
| `content_hash` | `str` | Yes | SHA-256 hash as 64 lowercase hex chars |
| `file_path` | `str \| None` | No | Local file path if applicable |

**Important**: `content_hash` must be exactly 64 lowercase hexadecimal characters. Use:
```python
import hashlib
content_hash = hashlib.sha256(content.encode()).hexdigest()
```

### Chunk Format (Chunking)

Chunking plugins must return chunks matching `ChunkDict`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | `str` | Yes | Chunk text content |
| `metadata` | `dict` | Yes | Chunk metadata (see below) |
| `chunk_id` | `str \| None` | No | Unique chunk identifier |
| `embedding` | `list[float] \| None` | No | Pre-computed embedding vector |

Metadata fields (all optional):
- `chunk_index`: Position in document
- `start_offset`, `end_offset`: Character positions
- `token_count`: Number of tokens
- `heading_hierarchy`: List of parent headings

### Rerank Result Format

Rerankers must return results matching `RerankResultDict`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `index` | `int` | Yes | Original document index |
| `score` | `float` | Yes | Relevance score (higher = better) |
| `text` | `str` | No | Document text |
| `metadata` | `dict` | No | Associated metadata |

---

## Valid String Constants

Since external plugins can't import semantik enums, use these string values:

### EMBEDDING_MODES
For `embed_texts()` mode parameter:
- `query` - Processing search queries (may apply prefixes)
- `document` - Processing documents for indexing

### EXTRACTION_TYPES
For `extract()` extraction_types parameter:
- `entities` - Named entity extraction
- `keywords` - Keyword extraction
- `language` - Language detection
- `topics` - Topic identification
- `sentiment` - Sentiment analysis
- `summary` - Text summarization
- `custom` - Custom extraction

---

## Testing Your Plugin

### Manual Verification

```bash
# Install your plugin
pip install -e ./my-plugin

# Verify it loads
python -c "
from my_plugin import MyConnector
print('PLUGIN_ID:', MyConnector.PLUGIN_ID)
print('PLUGIN_TYPE:', MyConnector.PLUGIN_TYPE)
print('Manifest:', MyConnector.get_manifest())
"
```

### Using Protocol Test Mixins

If you want to run the same tests semantik uses internally, you can install `semantik-shared` as a test dependency:

```python
# tests/test_my_connector.py
import pytest
from my_plugin import MyConnector

# Optional: use semantik's test mixins
try:
    from shared.plugins.testing.contracts import ConnectorProtocolTestMixin

    class TestMyConnectorProtocol(ConnectorProtocolTestMixin):
        plugin_class = MyConnector
except ImportError:
    pass  # semantik not installed, skip protocol tests


class TestMyConnector:
    """Tests that don't require semantik."""

    def test_has_required_attributes(self):
        assert hasattr(MyConnector, "PLUGIN_ID")
        assert hasattr(MyConnector, "PLUGIN_TYPE")
        assert hasattr(MyConnector, "PLUGIN_VERSION")
        assert MyConnector.PLUGIN_TYPE == "connector"

    def test_has_required_methods(self):
        assert callable(getattr(MyConnector, "authenticate", None))
        assert callable(getattr(MyConnector, "load_documents", None))
        assert callable(getattr(MyConnector, "get_manifest", None))

    @pytest.mark.asyncio
    async def test_load_documents_yields_valid_format(self):
        connector = MyConnector(config={"api_key": "test"})
        async for doc in connector.load_documents():
            assert "content" in doc
            assert "unique_id" in doc
            assert "source_type" in doc
            assert "metadata" in doc
            assert "content_hash" in doc
            assert len(doc["content_hash"]) == 64
            break
```

### Integration Testing with Semantik

```bash
# In a semantik development environment
python -c "
from shared.plugins.loader import _satisfies_protocol
from my_plugin import MyConnector

# Verify loader accepts your plugin
assert _satisfies_protocol(MyConnector, 'connector')
print('Plugin passes loader validation!')
"
```

---

## Packaging and Distribution

### pyproject.toml Template

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "semantik-plugin-myconnector"
version = "1.0.0"
description = "My custom connector for Semantik"
authors = [{name = "Your Name", email = "you@example.com"}]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    # Add your dependencies here
    # Do NOT add semantik as a dependency!
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[project.entry-points."semantik.plugins"]
my-connector = "my_connector.plugin:MyConnector"

[project.urls]
Homepage = "https://github.com/you/semantik-plugin-myconnector"
Issues = "https://github.com/you/semantik-plugin-myconnector/issues"
```

### Entry Point Format

The entry point format is:
```
plugin-id = "module.path:ClassName"
```

- `plugin-id`: Should match your `PLUGIN_ID` class variable
- `module.path`: Python import path to your module
- `ClassName`: Your plugin class name

### Distribution

**Via Git:**
```bash
pip install git+https://github.com/you/semantik-plugin-myconnector.git
```

**Via PyPI:**
```bash
pip install semantik-plugin-myconnector
```

---

## Troubleshooting

### Plugin Not Loading

1. **Check entry point registration:**
   ```bash
   pip show semantik-plugin-myconnector | grep -A 10 "Entry-points"
   ```

2. **Verify plugin type:**
   ```python
   assert MyConnector.PLUGIN_TYPE in ["connector", "embedding", "chunking", "reranker", "extractor", "agent"]
   ```

3. **Check for import errors:**
   ```python
   try:
       from my_plugin import MyConnector
   except ImportError as e:
       print(f"Import error: {e}")
   ```

### Validation Errors

If semantik rejects your plugin output, check:

1. **Missing required fields:**
   ```python
   # IngestedDocumentDict requires these fields
   required = {"content", "unique_id", "source_type", "metadata", "content_hash"}
   assert required.issubset(doc.keys())
   ```

2. **Invalid content_hash:**
   ```python
   assert len(doc["content_hash"]) == 64
   assert all(c in "0123456789abcdef" for c in doc["content_hash"])
   ```

3. **Invalid enum strings:**
   ```python
   # For agents
   assert msg["role"] in {"user", "assistant", "system", "tool_call", "tool_result", "error"}
   assert msg["type"] in {"text", "thinking", "tool_use", "tool_output", "partial", "final", "error", "metadata"}
   ```

### Debugging Tips

1. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test in isolation first:**
   ```python
   # Test your plugin without semantik
   import asyncio

   async def test():
       connector = MyConnector(config={"api_key": "test"})
       assert await connector.authenticate()
       async for doc in connector.load_documents():
           print(doc)
           break

   asyncio.run(test())
   ```

---

## Migration from ABC-based Plugins

If you have an existing plugin that inherits from semantik base classes, you can migrate to protocol-based:

1. Remove the import and inheritance:
   ```python
   # Before
   from shared.connectors.base import BaseConnector
   class MyConnector(BaseConnector):
       ...

   # After
   class MyConnector:  # No inheritance
       ...
   ```

2. Add the required class variables:
   ```python
   PLUGIN_ID: ClassVar[str] = "my-connector"
   PLUGIN_TYPE: ClassVar[str] = "connector"
   PLUGIN_VERSION: ClassVar[str] = "1.0.0"
   METADATA: ClassVar[dict[str, Any]] = {...}
   ```

3. Ensure all required methods return dict instead of dataclass:
   ```python
   # Before (returns IngestedDocument dataclass)
   yield IngestedDocument(content=..., unique_id=..., ...)

   # After (returns dict)
   yield {"content": ..., "unique_id": ..., ...}
   ```

4. Remove `semantik-shared` from your dependencies

The existing ABC-based approach still works. You only need to migrate if you want to:
- Avoid the semantik dependency
- Reduce import complexity
- Simplify distribution

---

## See Also

- [Plugin Protocols Reference](plugin-protocols.md) - Full protocol specifications
- [Plugin Development Guide](PLUGIN_DEVELOPMENT.md) - ABC-based plugin development
- [Plugin Testing](PLUGIN_TESTING.md) - Testing infrastructure
