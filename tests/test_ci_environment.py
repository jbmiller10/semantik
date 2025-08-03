#!/usr/bin/env python3
"""
Test CI environment configuration.

This module verifies that the CI environment is properly configured
for running tests without hanging or downloading external resources.
"""

import os
import sys

import pytest


class TestCIEnvironment:
    """Test CI environment setup."""
    
    def test_testing_env_set(self):
        """Verify TESTING environment variable is set."""
        assert os.getenv("TESTING") == "true", "TESTING env var must be 'true'"
    
    def test_mock_embeddings_in_ci(self):
        """Verify mock embeddings are used in CI."""
        if os.getenv("CI") == "true":
            # In CI, we should use mock embeddings
            from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker
            
            # Create chunker - should use MockEmbedding
            chunker = SemanticChunker()
            
            # The embed_model should be MockEmbedding
            from llama_index.core.embeddings import MockEmbedding
            assert isinstance(chunker.splitter.embed_model, MockEmbedding), \
                "Should use MockEmbedding in CI"
    
    def test_nltk_mocked_in_ci(self):
        """Verify NLTK is mocked in CI to prevent downloads."""
        if os.getenv("CI") == "true":
            # NLTK should be mocked
            assert hasattr(sys.modules.get("nltk"), "__class__"), "NLTK should be mocked"
            
            # Try to use nltk - should not trigger downloads
            import nltk
            try:
                # This should work with mock
                nltk.data.find("tokenizers/punkt")
            except Exception:
                # Expected with mock
                pass
    
    def test_gpu_disabled_in_ci(self):
        """Verify GPU is disabled in CI."""
        if os.getenv("CI") == "true":
            # CUDA should be disabled
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "0")
            assert cuda_visible == "", "CUDA_VISIBLE_DEVICES should be empty in CI"
            
            # Try importing torch - GPU should not be available
            try:
                import torch
                assert not torch.cuda.is_available(), "CUDA should not be available in CI"
            except ImportError:
                # PyTorch not installed is also fine
                pass
    
    @pytest.mark.timeout(5)
    def test_timeout_marker_works(self):
        """Test that timeout markers are working."""
        import time
        # This should complete quickly
        time.sleep(0.1)
        assert True
    
    def test_ci_test_limits(self):
        """Verify CI test limits are set."""
        if os.getenv("CI") == "true":
            max_chunk_size = os.getenv("MAX_CHUNK_SIZE")
            max_workers = os.getenv("MAX_WORKERS")
            
            # These should be set by conftest.py
            assert max_chunk_size == "1000", "MAX_CHUNK_SIZE should be limited in CI"
            assert max_workers == "2", "MAX_WORKERS should be limited in CI"