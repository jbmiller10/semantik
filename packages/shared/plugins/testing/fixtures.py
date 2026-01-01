"""Pytest fixtures for plugin testing.

Import these fixtures in your plugin's conftest.py:

    from shared.plugins.testing.fixtures import *

Or import specific fixtures:

    from shared.plugins.testing.fixtures import sample_text, sample_documents
"""

from __future__ import annotations

import pytest

from .mocks import (
    MockChunker,
    MockDocument,
    MockEmbeddingService,
    MockExtractor,
    MockReranker,
)

# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture()
def sample_text() -> str:
    """Sample text for testing extractors and chunkers."""
    return (
        "Apple Inc. is an American multinational technology company headquartered "
        "in Cupertino, California. The company was founded by Steve Jobs, Steve Wozniak, "
        "and Ronald Wayne in 1976. Apple designs, develops, and sells consumer electronics, "
        "computer software, and online services."
    )


@pytest.fixture()
def sample_short_text() -> str:
    """Short sample text for quick tests."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture()
def sample_documents() -> list[str]:
    """Sample documents for testing rerankers and batch processing."""
    return [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Deep learning uses neural networks with many layers to learn complex patterns in data.",
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
    ]


@pytest.fixture()
def sample_query() -> str:
    """Sample search query for testing rerankers."""
    return "How do machines learn from data?"


@pytest.fixture()
def sample_long_document() -> str:
    """Long document for testing chunking strategies."""
    paragraphs = [
        "Artificial intelligence (AI) has rapidly transformed numerous industries over the past decade.",
        "In healthcare, AI systems assist doctors in diagnosing diseases and recommending treatments.",
        "Financial institutions use machine learning to detect fraud and assess credit risk.",
        "Autonomous vehicles rely on computer vision and sensor fusion to navigate safely.",
        "Natural language models have revolutionized how we interact with technology through chatbots.",
        "The ethical implications of AI continue to be debated as systems become more powerful.",
        "Researchers are working on explainable AI to make model decisions more transparent.",
        "Edge computing enables AI inference to run on devices without cloud connectivity.",
        "Transfer learning allows models trained on one task to be adapted for related tasks.",
        "The future of AI promises even more profound changes to society and the economy.",
    ]
    return "\n\n".join(paragraphs)


@pytest.fixture()
def sample_metadata() -> dict:
    """Sample metadata dictionary for testing."""
    return {
        "source": "test",
        "author": "Test Author",
        "date": "2025-01-01",
        "tags": ["ai", "testing", "sample"],
    }


@pytest.fixture()
def sample_mock_document(sample_text: str, sample_metadata: dict) -> MockDocument:
    """Sample MockDocument for testing connectors."""
    return MockDocument(
        content=sample_text,
        source_id="test-source-123",
        file_path="documents/sample.txt",
        metadata=sample_metadata,
    )


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture()
def mock_embedding_service() -> MockEmbeddingService:
    """Mock embedding service with default dimension."""
    return MockEmbeddingService(dimension=384)


@pytest.fixture()
def mock_embedding_service_768() -> MockEmbeddingService:
    """Mock embedding service with 768 dimension (common for BERT-based models)."""
    return MockEmbeddingService(dimension=768)


@pytest.fixture()
def mock_reranker() -> MockReranker:
    """Mock reranker for testing."""
    return MockReranker()


@pytest.fixture()
def mock_extractor() -> MockExtractor:
    """Mock extractor for testing."""
    return MockExtractor()


@pytest.fixture()
def mock_chunker() -> MockChunker:
    """Mock chunker with default settings."""
    return MockChunker(chunk_size=100, overlap=20)


@pytest.fixture()
def mock_chunker_large() -> MockChunker:
    """Mock chunker with larger chunks."""
    return MockChunker(chunk_size=500, overlap=50)


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture()
def sample_plugin_config() -> dict:
    """Sample plugin configuration for testing."""
    return {
        "api_key_env": "TEST_API_KEY",
        "model": "test-model",
        "batch_size": 32,
        "timeout": 30,
    }


@pytest.fixture()
def sample_embedding_config() -> dict:
    """Sample embedding plugin configuration."""
    return {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
        "normalize": True,
    }


@pytest.fixture()
def sample_reranker_config() -> dict:
    """Sample reranker plugin configuration."""
    return {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "max_documents": 100,
        "device": "cpu",
    }


@pytest.fixture()
def sample_extractor_config() -> dict:
    """Sample extractor plugin configuration."""
    return {
        "model_name": "en_core_web_sm",
        "extract_entities": True,
        "extract_keywords": True,
        "max_keywords": 10,
    }


@pytest.fixture()
def sample_chunk_config() -> dict:
    """Sample chunking configuration."""
    return {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "min_chunk_size": 100,
    }


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture()
def _env_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment with a test API key."""
    monkeypatch.setenv("TEST_API_KEY", "test-key-12345")


@pytest.fixture()
def _env_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up environment without API key (for testing missing config)."""
    monkeypatch.delenv("TEST_API_KEY", raising=False)
