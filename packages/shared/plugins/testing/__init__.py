"""Plugin testing utilities for Semantik.

This module provides pytest fixtures and contract test base classes
for testing Semantik plugins.

Quick Start
-----------

1. Create a test file for your plugin:

    ```python
    # tests/test_my_plugin.py
    import pytest
    from shared.plugins.testing import (
        EmbeddingPluginContractTest,
        sample_text,
        mock_embedding_service,
    )
    from my_plugin import MyEmbeddingPlugin

    class TestMyEmbeddingPlugin(EmbeddingPluginContractTest):
        plugin_class = MyEmbeddingPlugin

        @pytest.fixture
        def plugin_config(self):
            return {"model_name": "my-model"}

        # Contract tests run automatically!
        # Add custom tests below:

        def test_my_custom_feature(self, plugin_instance, sample_text):
            result = plugin_instance.process(sample_text)
            assert result is not None
    ```

2. Run tests:

    ```bash
    pytest tests/test_my_plugin.py -v
    ```

Contract Test Classes
---------------------

- ``PluginContractTest``: Base class for all plugin types
- ``EmbeddingPluginContractTest``: For embedding providers
- ``ChunkingPluginContractTest``: For chunking strategies
- ``ConnectorPluginContractTest``: For data source connectors
- ``RerankerPluginContractTest``: For document rerankers
- ``ExtractorPluginContractTest``: For metadata extractors

Fixtures
--------

Sample data fixtures:
- ``sample_text``: Sample text for testing
- ``sample_short_text``: Short text for quick tests
- ``sample_documents``: List of documents for batch tests
- ``sample_query``: Sample search query
- ``sample_long_document``: Long document for chunking tests
- ``sample_metadata``: Sample metadata dict
- ``sample_mock_document``: MockDocument instance

Mock service fixtures:
- ``mock_embedding_service``: MockEmbeddingService (384d)
- ``mock_embedding_service_768``: MockEmbeddingService (768d)
- ``mock_reranker``: MockReranker
- ``mock_extractor``: MockExtractor
- ``mock_chunker``: MockChunker

Configuration fixtures:
- ``sample_plugin_config``: Generic plugin config
- ``sample_embedding_config``: Embedding config
- ``sample_reranker_config``: Reranker config
- ``sample_extractor_config``: Extractor config
- ``sample_chunk_config``: Chunking config

Environment fixtures:
- ``env_with_api_key``: Sets TEST_API_KEY in environment
- ``env_without_api_key``: Removes TEST_API_KEY from environment

Mock Classes
------------

- ``MockDocument``: Mock document for testing
- ``MockEmbeddingService``: Mock embedding service
- ``MockReranker``: Mock reranker
- ``MockExtractor``: Mock extractor
- ``MockChunker``: Mock chunker
"""

from .contracts import (
    ChunkingPluginContractTest,
    ConnectorPluginContractTest,
    EmbeddingPluginContractTest,
    ExtractorPluginContractTest,
    PluginContractTest,
    RerankerPluginContractTest,
)
from .fixtures import (
    _env_with_api_key,
    _env_without_api_key,
    mock_chunker,
    mock_chunker_large,
    mock_embedding_service,
    mock_embedding_service_768,
    mock_extractor,
    mock_reranker,
    sample_chunk_config,
    sample_documents,
    sample_embedding_config,
    sample_extractor_config,
    sample_long_document,
    sample_metadata,
    sample_mock_document,
    sample_plugin_config,
    sample_query,
    sample_reranker_config,
    sample_short_text,
    sample_text,
)
from .mocks import (
    MockChunker,
    MockDocument,
    MockEmbeddingService,
    MockExtractor,
    MockReranker,
)

__all__ = [
    # Contract test classes
    "PluginContractTest",
    "EmbeddingPluginContractTest",
    "ChunkingPluginContractTest",
    "ConnectorPluginContractTest",
    "RerankerPluginContractTest",
    "ExtractorPluginContractTest",
    # Mock classes
    "MockDocument",
    "MockEmbeddingService",
    "MockReranker",
    "MockExtractor",
    "MockChunker",
    # Sample data fixtures
    "sample_text",
    "sample_short_text",
    "sample_documents",
    "sample_query",
    "sample_long_document",
    "sample_metadata",
    "sample_mock_document",
    # Mock service fixtures
    "mock_embedding_service",
    "mock_embedding_service_768",
    "mock_reranker",
    "mock_extractor",
    "mock_chunker",
    "mock_chunker_large",
    # Config fixtures
    "sample_plugin_config",
    "sample_embedding_config",
    "sample_reranker_config",
    "sample_extractor_config",
    "sample_chunk_config",
    # Environment fixtures
    "_env_with_api_key",
    "_env_without_api_key",
]
