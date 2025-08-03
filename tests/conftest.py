#!/usr/bin/env python3
"""
Pytest configuration for test suite.

This module configures the test environment, including mocking external
dependencies and preventing network requests in CI.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure TESTING environment is set before any imports
os.environ["TESTING"] = "true"

# Detect CI environment
IS_CI = os.getenv("CI", "false").lower() == "true"

# Mock NLTK to prevent downloads in CI
if IS_CI:
    # Create comprehensive NLTK mock
    nltk_mock = MagicMock()
    nltk_mock.__version__ = "3.8"
    
    # Mock all NLTK submodules that might be imported
    sys.modules["nltk"] = nltk_mock
    sys.modules["nltk.data"] = MagicMock()
    sys.modules["nltk.tokenize"] = MagicMock()
    sys.modules["nltk.corpus"] = MagicMock()
    sys.modules["nltk.download"] = MagicMock()
    
    # Mock sentence splitter to return basic splits
    def mock_sent_tokenize(text):
        """Simple sentence tokenizer for tests."""
        if not text:
            return []
        # Basic sentence splitting on periods
        sentences = []
        for s in text.split("."):
            s = s.strip()
            if s:
                sentences.append(s + ".")
        return sentences if sentences else [text]
    
    sys.modules["nltk.tokenize"].sent_tokenize = mock_sent_tokenize
    sys.modules["nltk"].tokenize = sys.modules["nltk.tokenize"]
    
    # Mock punkt tokenizer
    punkt_mock = MagicMock()
    punkt_mock.tokenize = mock_sent_tokenize
    sys.modules["nltk.tokenize"].punkt = MagicMock()
    sys.modules["nltk.tokenize"].punkt.PunktSentenceTokenizer = MagicMock(return_value=punkt_mock)


@pytest.fixture(scope="session", autouse=True)
def ensure_test_environment():
    """Ensure test environment is properly configured."""
    # Verify TESTING is set
    assert os.getenv("TESTING") == "true", "TESTING environment variable must be set"
    
    # In CI, ensure we're using mock embeddings
    if IS_CI:
        assert os.getenv("USE_MOCK_EMBEDDINGS") == "true", "USE_MOCK_EMBEDDINGS must be true in CI"
        
        # Disable GPU in CI
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set small limits for CI
        os.environ["MAX_CHUNK_SIZE"] = "1000"
        os.environ["MAX_WORKERS"] = "2"


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test."""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clean up any GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU as unavailable for testing CPU paths."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def ci_environment():
    """Fixture that sets up CI-specific test configuration."""
    original_env = os.environ.copy()
    
    # Set CI-specific environment
    os.environ["CI"] = "true"
    os.environ["TESTING"] = "true"
    os.environ["USE_MOCK_EMBEDDINGS"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Configure pytest timeout defaults
def pytest_configure(config):
    """Configure pytest with timeout defaults."""
    # Set default timeout for all tests in CI
    if IS_CI and hasattr(config.option, 'timeout'):
        if not config.option.timeout:
            config.option.timeout = 60  # 60 seconds default timeout in CI
            config.option.timeout_method = "thread"


# Skip certain test categories in CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    if not IS_CI:
        return
    
    skip_ci = pytest.mark.skip(reason="Skipped in CI environment")
    
    for item in items:
        # Skip GPU-intensive tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(skip_ci)
        
        # Skip tests that require large models
        if "large_model" in item.keywords:
            item.add_marker(skip_ci)
        
        # Skip tests that download data
        if "downloads_data" in item.keywords:
            item.add_marker(skip_ci)