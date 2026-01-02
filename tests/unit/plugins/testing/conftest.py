"""Pytest configuration for plugin testing tests."""

from __future__ import annotations

# Explicitly import fixtures to override any from parent conftest.py
# These fixtures are specifically for testing the plugin testing module itself
from shared.plugins.testing.fixtures import (
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

__all__ = [
    "mock_chunker",
    "mock_chunker_large",
    "mock_embedding_service",
    "mock_embedding_service_768",
    "mock_extractor",
    "mock_reranker",
    "sample_chunk_config",
    "sample_documents",
    "sample_embedding_config",
    "sample_extractor_config",
    "sample_long_document",
    "sample_metadata",
    "sample_mock_document",
    "sample_plugin_config",
    "sample_query",
    "sample_reranker_config",
    "sample_short_text",
    "sample_text",
]
