# Plugin Testing Guide

This guide explains how to test Semantik plugins using the built-in testing utilities.

## Overview

Semantik provides a testing framework that helps plugin authors:

1. **Verify contract compliance** - Ensure your plugin implements all required methods
2. **Use standardized fixtures** - Common test data and mock services
3. **Write isolated tests** - Mock dependencies without real services

## Quick Start

### 1. Install Test Dependencies

```bash
pip install pytest pytest-asyncio
```

### 2. Create a Test File

```python
# tests/test_my_plugin.py
import pytest
from shared.plugins.testing import EmbeddingPluginContractTest
from my_plugin import MyEmbeddingPlugin


class TestMyEmbeddingPlugin(EmbeddingPluginContractTest):
    """Test suite for MyEmbeddingPlugin.

    Inheriting from EmbeddingPluginContractTest automatically runs
    all contract verification tests.
    """

    plugin_class = MyEmbeddingPlugin

    @pytest.fixture
    def plugin_config(self):
        """Provide configuration for tests."""
        return {
            "model_name": "my-model",
            "batch_size": 32,
        }

    # Contract tests run automatically!

    # Add custom tests below:
    def test_my_custom_feature(self, plugin_instance):
        """Test a custom feature of your plugin."""
        assert plugin_instance.supports_feature_x()
```

### 3. Run Tests

```bash
pytest tests/test_my_plugin.py -v
```

## Contract Test Classes

### PluginContractTest (Base)

All plugins must pass these tests:

| Test | Description |
|------|-------------|
| `test_has_plugin_type` | PLUGIN_TYPE class attribute exists |
| `test_has_plugin_id` | PLUGIN_ID class attribute exists |
| `test_has_plugin_version` | PLUGIN_VERSION follows semver |
| `test_get_manifest_returns_valid_manifest` | get_manifest() returns valid PluginManifest |
| `test_get_config_schema_returns_valid_schema_or_none` | Config schema is valid or None |
| `test_health_check_returns_bool` | health_check() returns boolean |
| `test_initialize_and_cleanup` | Lifecycle methods work |
| `test_config_property` | Config property accessible |

### EmbeddingPluginContractTest

Additional tests for embedding plugins:

| Test | Description |
|------|-------------|
| `test_plugin_type_is_embedding` | PLUGIN_TYPE == "embedding" |
| `test_has_embed_single_method` | Has embed_single() method |
| `test_has_embed_texts_method` | Has embed_texts() method |
| `test_has_get_dimension_method` | Has get_dimension() method |

### RerankerPluginContractTest

Additional tests for reranker plugins:

| Test | Description |
|------|-------------|
| `test_plugin_type_is_reranker` | PLUGIN_TYPE == "reranker" |
| `test_has_rerank_method` | Has rerank() method |
| `test_has_get_capabilities_classmethod` | Has get_capabilities() |
| `test_get_capabilities_returns_valid_capabilities` | Returns valid RerankerCapabilities |

### ExtractorPluginContractTest

Additional tests for extractor plugins:

| Test | Description |
|------|-------------|
| `test_plugin_type_is_extractor` | PLUGIN_TYPE == "extractor" |
| `test_supported_extractions_returns_list` | Returns list of ExtractionType |
| `test_has_extract_method` | Has extract() method |
| `test_extract_returns_extraction_result` | Returns valid ExtractionResult |

### ConnectorPluginContractTest

Additional tests for connector plugins:

| Test | Description |
|------|-------------|
| `test_plugin_type_is_connector` | PLUGIN_TYPE == "connector" |
| `test_has_authenticate_method` | Has authenticate() method |
| `test_has_load_documents_method` | Has load_documents() method |
| `test_has_get_config_fields_classmethod` | Has get_config_fields() |
| `test_get_config_fields_returns_valid_fields` | Returns valid field definitions |

### ChunkingPluginContractTest

Additional tests for chunking plugins:

| Test | Description |
|------|-------------|
| `test_plugin_type_is_chunking` | PLUGIN_TYPE == "chunking" |
| `test_has_chunk_method` | Has chunk() method |
| `test_has_validate_content_method` | Has validate_content() method |
| `test_has_estimate_chunks_method` | Has estimate_chunks() method |

## Available Fixtures

### Sample Data

```python
@pytest.fixture
def sample_text():
    """Sample text about Apple Inc."""
    return "Apple Inc. is an American multinational..."

@pytest.fixture
def sample_documents():
    """List of 5 AI-related documents."""
    return ["Machine learning is...", "Natural language...", ...]

@pytest.fixture
def sample_query():
    """Sample search query."""
    return "How do machines learn from data?"

@pytest.fixture
def sample_long_document():
    """Long document with 10 paragraphs."""
    return "Artificial intelligence (AI)..."
```

### Mock Services

```python
@pytest.fixture
def mock_embedding_service():
    """Mock embedding service (384 dimensions)."""
    return MockEmbeddingService(dimension=384)

@pytest.fixture
def mock_reranker():
    """Mock reranker that scores by word overlap."""
    return MockReranker()

@pytest.fixture
def mock_extractor():
    """Mock extractor returning simple results."""
    return MockExtractor()
```

### Configuration

```python
@pytest.fixture
def sample_plugin_config():
    """Generic plugin configuration."""
    return {"api_key_env": "TEST_API_KEY", "model": "test-model", ...}

@pytest.fixture
def sample_embedding_config():
    """Embedding-specific configuration."""
    return {"model_name": "all-MiniLM-L6-v2", "device": "cpu", ...}
```

### Environment

```python
@pytest.fixture
def env_with_api_key(monkeypatch):
    """Set TEST_API_KEY in environment."""
    monkeypatch.setenv("TEST_API_KEY", "test-key-12345")

@pytest.fixture
def env_without_api_key(monkeypatch):
    """Remove TEST_API_KEY from environment."""
    monkeypatch.delenv("TEST_API_KEY", raising=False)
```

## Mock Classes

### MockEmbeddingService

```python
from shared.plugins.testing import MockEmbeddingService

# Create with custom dimension
service = MockEmbeddingService(dimension=768)

# Embed text (returns deterministic vectors based on text hash)
embedding = await service.embed_single("Hello world")
embeddings = await service.embed_texts(["Hello", "World"])

# Check what was called
assert service.embed_calls == ["Hello world"]
assert service.batch_calls == [["Hello", "World"]]
```

### MockReranker

```python
from shared.plugins.testing import MockReranker

reranker = MockReranker()

# Rerank documents (scores by word overlap with query)
results = await reranker.rerank(
    query="machine learning",
    documents=["ML is great", "Cooking recipes", "AI and ML"],
    top_k=2
)

# Returns [(original_index, score, document)]
assert len(results) == 2
```

### MockExtractor

```python
from shared.plugins.testing import MockExtractor

extractor = MockExtractor()

# Extract metadata (returns first 3 words as mock entities)
result = await extractor.extract("Apple announces new products")

assert result["entities"][0]["text"] == "Apple"
assert result["language"] == "en"
```

### MockDocument

```python
from shared.plugins.testing import MockDocument

doc = MockDocument(
    content="Document content here",
    source_id="source-123",
    file_path="docs/readme.md",
    metadata={"author": "Test"}
)
```

## Writing Custom Tests

### Async Tests

```python
import pytest

class TestMyPlugin(EmbeddingPluginContractTest):
    plugin_class = MyPlugin

    @pytest.mark.asyncio
    async def test_batch_embedding(self, plugin_instance, sample_documents):
        """Test batch embedding works correctly."""
        await plugin_instance.initialize()

        try:
            embeddings = await plugin_instance.embed_texts(sample_documents)

            assert len(embeddings) == len(sample_documents)
            assert all(len(e) == 384 for e in embeddings)
        finally:
            await plugin_instance.cleanup()
```

### Testing Configuration

```python
class TestMyPlugin(EmbeddingPluginContractTest):
    plugin_class = MyPlugin

    @pytest.fixture
    def plugin_config(self):
        return {"model_name": "custom-model"}

    def test_config_is_applied(self, plugin_instance):
        """Verify configuration was applied."""
        assert plugin_instance.config["model_name"] == "custom-model"
```

### Testing Health Checks

```python
class TestMyPlugin(ExtractorPluginContractTest):
    plugin_class = MyPlugin

    @pytest.mark.asyncio
    async def test_health_check_with_invalid_config(self):
        """Health check should return False with missing API key."""
        result = await self.plugin_class.health_check(config={})
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_with_valid_config(self, env_with_api_key):
        """Health check should return True with valid config."""
        result = await self.plugin_class.health_check(
            config={"api_key_env": "TEST_API_KEY"}
        )
        assert result is True
```

### Testing Error Handling

```python
class TestMyPlugin(RerankerPluginContractTest):
    plugin_class = MyPlugin

    @pytest.mark.asyncio
    async def test_handles_empty_documents(self, plugin_instance):
        """Plugin should handle empty document list gracefully."""
        await plugin_instance.initialize()

        try:
            results = await plugin_instance.rerank(
                query="test",
                documents=[],
                top_k=10
            )
            assert results == []
        finally:
            await plugin_instance.cleanup()
```

## Best Practices

### 1. Always Clean Up

```python
@pytest.mark.asyncio
async def test_something(self, plugin_instance):
    await plugin_instance.initialize()
    try:
        # Your test code
        pass
    finally:
        await plugin_instance.cleanup()  # Always clean up!
```

### 2. Use Fixtures for Common Setup

```python
@pytest.fixture
async def initialized_plugin(self, plugin_instance):
    """Provide an initialized plugin."""
    await plugin_instance.initialize()
    yield plugin_instance
    await plugin_instance.cleanup()

async def test_something(self, initialized_plugin):
    # Plugin is already initialized
    result = await initialized_plugin.do_something()
```

### 3. Test Edge Cases

```python
def test_empty_input(self, plugin_instance):
    result = plugin_instance.process("")
    assert result is not None

def test_very_long_input(self, plugin_instance, sample_long_document):
    # Should not raise
    result = plugin_instance.process(sample_long_document * 100)
```

### 4. Mock External Services

```python
def test_with_mock_embedding(self, plugin_instance, mock_embedding_service):
    """Use mock instead of real embedding service."""
    plugin_instance._embedding_service = mock_embedding_service

    result = plugin_instance.process_with_embeddings("test")

    assert mock_embedding_service.embed_calls == ["test"]
```

## Running Tests

### Run All Plugin Tests

```bash
pytest tests/ -v
```

### Run Contract Tests Only

```bash
pytest tests/ -v -k "contract"
```

### Run with Coverage

```bash
pytest tests/ --cov=my_plugin --cov-report=html
```

### Run Async Tests

```bash
pytest tests/ -v --asyncio-mode=auto
```

## Troubleshooting

### "Plugin class not set"

Make sure you set the `plugin_class` attribute:

```python
class TestMyPlugin(EmbeddingPluginContractTest):
    plugin_class = MyPlugin  # Required!
```

### "Async tests not running"

Add the pytest-asyncio marker:

```python
@pytest.mark.asyncio
async def test_async_method(self):
    ...
```

Or configure in `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
```

### "Fixture not found"

Import fixtures in your conftest.py:

```python
# tests/conftest.py
from shared.plugins.testing.fixtures import *
```
