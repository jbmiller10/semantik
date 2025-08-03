#!/usr/bin/env python3
"""
Critical privacy tests for semantic chunking.
These tests ensure NO data leaves the user's system during semantic chunking.
"""
import asyncio
import gc
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import logging
from unittest.mock import patch, MagicMock
import pytest
import requests

from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker

# Test sensitive content that should NEVER leave the system
SENSITIVE_TEST_CONTENT = [
    "Social Security Number: 123-45-6789",
    "Credit Card: 4532-1234-5678-9012",
    "API Key: sk-1234567890abcdef",
    "Password: supersecret123",
    "Personal Health Information: Patient John Doe has diabetes",
    "Confidential: Company earnings will be $10M this quarter",
]

@pytest.mark.security
@pytest.mark.privacy
class TestEmbeddingPrivacy:
    """Test that semantic chunking preserves data privacy."""
    
    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Ensure testing environment."""
        os.environ["TESTING"] = "false"  # Test production mode
        os.environ["ENABLE_OPENAI_EMBEDDINGS"] = "false"  # Disable external APIs
        
    def test_no_network_requests_during_semantic_chunking(self):
        """CRITICAL: Verify no network requests are made during semantic chunking."""
        # Mock all possible network libraries
        with patch('requests.get') as mock_requests_get, \
             patch('requests.post') as mock_requests_post, \
             patch('httpx.get') as mock_httpx_get, \
             patch('httpx.post') as mock_httpx_post, \
             patch('urllib.request.urlopen') as mock_urllib, \
             patch('urllib.request.Request') as mock_urllib_req, \
             patch('socket.create_connection') as mock_socket:
            
            # Process sensitive content with semantic chunking
            config = {
                "strategy": "semantic",
                "params": {
                    "breakpoint_percentile_threshold": 90,
                    "buffer_size": 1,
                    "max_chunk_size": 1000
                }
            }
            
            chunker = ChunkingFactory.create_chunker(config)
            
            for content in SENSITIVE_TEST_CONTENT:
                chunks = chunker.chunk_text(content, "privacy_test")
                assert len(chunks) >= 1  # Should produce chunks
            
            # CRITICAL: Assert NO network calls were made
            mock_requests_get.assert_not_called()
            mock_requests_post.assert_not_called()
            mock_httpx_get.assert_not_called()
            mock_httpx_post.assert_not_called()
            mock_urllib.assert_not_called()
            mock_urllib_req.assert_not_called()
            mock_socket.assert_not_called()
    
    async def test_no_network_requests_async_semantic_chunking(self):
        """CRITICAL: Verify no network requests in async semantic chunking."""
        # Test async version with same network isolation
        with patch('aiohttp.ClientSession.get') as mock_aiohttp_get, \
             patch('aiohttp.ClientSession.post') as mock_aiohttp_post, \
             patch('httpx.AsyncClient.get') as mock_httpx_async_get, \
             patch('httpx.AsyncClient.post') as mock_httpx_async_post:
            
            config = {"strategy": "semantic", "params": {}}
            chunker = ChunkingFactory.create_chunker(config)
            
            for content in SENSITIVE_TEST_CONTENT:
                chunks = await chunker.chunk_text_async(content, "async_privacy_test")
                assert len(chunks) >= 1
            
            # Assert no async network calls
            mock_aiohttp_get.assert_not_called()
            mock_aiohttp_post.assert_not_called()
            mock_httpx_async_get.assert_not_called()
            mock_httpx_async_post.assert_not_called()
    
    def test_local_embedding_model_verification(self):
        """Verify that only local embedding models are loaded."""
        config = {"strategy": "semantic", "params": {}}
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
            
            # Mock successful local model initialization
            mock_hf.return_value = MagicMock()
            
            chunker = ChunkingFactory.create_chunker(config)
            
            # Verify local models were attempted to be loaded
            mock_hf.assert_called_once()
            
            # Verify model parameters are for local processing
            call_kwargs = mock_hf.call_args.kwargs
            assert 'model_name' in call_kwargs
            assert call_kwargs['device'] in ['cpu', 'cuda']  # Local devices only
            assert 'cache_folder' in call_kwargs  # Local cache
    
    def test_openai_embeddings_properly_disabled(self):
        """Test that OpenAI embeddings are disabled by default."""
        config = {"strategy": "semantic", "params": {}}
        
        # Force local embedding failure to test fallback
        with patch('sentence_transformers.SentenceTransformer', side_effect=ImportError("No local model")), \
             patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding', side_effect=ImportError("No HF")):
            
            # Should raise error, not fall back to OpenAI
            with pytest.raises(RuntimeError, match="Cannot initialize semantic chunking"):
                ChunkingFactory.create_chunker(config)
    
    def test_model_files_are_local(self):
        """Verify embedding models use local files only."""
        config = {"strategy": "semantic", "params": {}}
        
        with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
            mock_instance = MagicMock()
            mock_hf.return_value = mock_instance
            
            chunker = ChunkingFactory.create_chunker(config)
            
            # Check that cache_folder is set to local directory
            call_kwargs = mock_hf.call_args.kwargs
            cache_folder = call_kwargs.get('cache_folder', './models')
            
            # Should be local path, not URL
            assert not cache_folder.startswith('http')
            assert not cache_folder.startswith('https')
            assert cache_folder.startswith('.') or cache_folder.startswith('/')
    
    def test_embedding_model_inference_is_local(self):
        """Test that embedding inference happens locally."""
        config = {"strategy": "semantic", "params": {}}
        
        # Mock a working local embedding model
        with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
            mock_embedding_model = MagicMock()
            mock_hf.return_value = mock_embedding_model
            
            # Mock the embedding generation
            mock_embedding_model.get_text_embedding.return_value = [0.1] * 384
            
            chunker = ChunkingFactory.create_chunker(config)
            chunks = chunker.chunk_text("Test content", "local_inference_test")
            
            # Verify embedding was called locally
            assert mock_embedding_model.get_text_embedding.called
            
            # Verify no external API calls during inference
            with patch('requests.post') as mock_post:
                # Process more content to trigger embeddings
                chunker.chunk_text("More test content for embeddings", "inference_test_2")
                mock_post.assert_not_called()

    def test_no_data_in_logs_or_cache(self):
        """Ensure sensitive data doesn't appear in logs or cache files."""
        import tempfile
        import logging
        from io import StringIO
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('packages.shared.text_processing')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        try:
            config = {"strategy": "semantic", "params": {}}
            chunker = ChunkingFactory.create_chunker(config)
            
            # Process sensitive content
            sensitive_content = "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012"
            chunker.chunk_text(sensitive_content, "leak_test")
            
            # Check that sensitive data is not in logs
            log_output = log_capture.getvalue()
            assert "123-45-6789" not in log_output
            assert "4532-1234-5678-9012" not in log_output
            
        finally:
            logger.removeHandler(handler)
    
    def test_concurrent_processing_isolation(self):
        """Test that concurrent chunking operations don't leak data between requests."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        config = {"strategy": "semantic", "params": {}}
        
        def process_sensitive_content(content_id):
            chunker = ChunkingFactory.create_chunker(config)
            sensitive_content = f"Secret {content_id}: {content_id * 'sensitive_data'}"
            return chunker.chunk_text(sensitive_content, f"concurrent_test_{content_id}")
        
        # Process multiple sensitive documents concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_sensitive_content, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # Verify each result only contains its own content
        for i, chunks in enumerate(results):
            combined_text = " ".join(chunk.text for chunk in chunks)
            
            # Should contain own content
            assert f"Secret {i}" in combined_text
            
            # Should NOT contain other requests' content
            for j in range(10):
                if i != j:
                    assert f"Secret {j}" not in combined_text
    
    def test_memory_cleanup_prevents_leaks(self):
        """Test that memory is properly cleaned up to prevent data leaks."""
        import gc
        
        config = {"strategy": "semantic", "params": {}}
        chunker = ChunkingFactory.create_chunker(config)
        
        # Process sensitive content
        sensitive_content = "Highly confidential information that must not leak"
        chunks = chunker.chunk_text(sensitive_content, "memory_test")
        
        # Clear references
        del chunks
        del chunker
        
        # Force garbage collection
        gc.collect()
        
        # Verify sensitive content is not in memory
        # This is a best-effort check - can't guarantee complete memory scanning
        for obj in gc.get_objects():
            if isinstance(obj, str) and "Highly confidential information" in obj:
                pytest.fail("Sensitive content found in memory after cleanup")

    def test_insecure_configuration_blocked(self):
        """Test that insecure configurations are blocked."""
        # Test that OpenAI embeddings require explicit enabling
        os.environ["ENABLE_OPENAI_EMBEDDINGS"] = "false"
        
        config = {
            "strategy": "semantic",
            "params": {
                "embed_model": "openai"  # Should be blocked
            }
        }
        
        with pytest.raises((ValueError, RuntimeError)):
            ChunkingFactory.create_chunker(config)
    
    def test_production_mode_privacy_enforcement(self):
        """Test that production mode enforces privacy settings."""
        os.environ["TESTING"] = "false"  # Production mode
        os.environ["ENABLE_OPENAI_EMBEDDINGS"] = "true"  # Even if enabled
        
        # Should still prefer local models
        config = {"strategy": "semantic", "params": {}}
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
            
            mock_hf.return_value = MagicMock()
            chunker = ChunkingFactory.create_chunker(config)
            
            # Should attempt local models first
            mock_hf.assert_called_once()
    
    def test_openai_fallback_with_explicit_warning(self):
        """Test that OpenAI fallback warns clearly when enabled."""
        os.environ["TESTING"] = "false"
        os.environ["ENABLE_OPENAI_EMBEDDINGS"] = "true"
        
        config = {"strategy": "semantic", "params": {}}
        
        # Mock local embedding failure
        with patch('sentence_transformers.SentenceTransformer', side_effect=ImportError("No local")), \
             patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding', side_effect=ImportError("No HF")), \
             patch('llama_index.embeddings.openai.OpenAIEmbedding') as mock_openai:
            
            mock_openai.return_value = MagicMock()
            
            # Capture logs
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger = logging.getLogger('packages.shared.text_processing')
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            
            try:
                chunker = ChunkingFactory.create_chunker(config)
                
                # Should have warning about data leaving system
                log_output = log_capture.getvalue()
                assert "DATA WILL LEAVE YOUR SYSTEM" in log_output, "Missing privacy warning for OpenAI embeddings"
            finally:
                logger.removeHandler(handler)
    
    def test_embedding_api_request_interception(self):
        """Test that we can intercept and block any embedding API requests."""
        config = {"strategy": "semantic", "params": {}}
        
        # Create a custom interceptor for any HTTP requests
        original_request = None
        if hasattr(requests, 'request'):
            original_request = requests.request
        
        def intercepted_request(*args, **kwargs):
            url = kwargs.get('url', args[1] if len(args) > 1 else '')
            if 'openai' in str(url).lower() or 'api' in str(url).lower():
                raise AssertionError(f"Detected external API request to: {url}")
            return original_request(*args, **kwargs) if original_request else None
        
        with patch('requests.request', side_effect=intercepted_request):
            with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
                mock_hf.return_value = MagicMock()
                
                chunker = ChunkingFactory.create_chunker(config)
                
                # Process sensitive content
                for content in SENSITIVE_TEST_CONTENT:
                    chunks = chunker.chunk_text(content, "api_intercept_test")
                    assert len(chunks) >= 1
    
    def test_different_embedding_models_are_local(self):
        """Test various embedding models are all local."""
        models_to_test = [
            "all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-mpnet-base-v2",
        ]
        
        for model_name in models_to_test:
            os.environ["EMBEDDING_MODEL"] = model_name
            config = {"strategy": "semantic", "params": {}}
            
            with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
                mock_hf.return_value = MagicMock()
                
                chunker = ChunkingFactory.create_chunker(config)
                
                # Verify model is loaded locally
                call_kwargs = mock_hf.call_args.kwargs
                assert call_kwargs['model_name'] == model_name
                assert 'cache_folder' in call_kwargs
                assert call_kwargs['device'] in ['cpu', 'cuda']
    
    @pytest.mark.parametrize("gpu_available", [True, False])
    def test_gpu_and_cpu_configurations_are_local(self, gpu_available):
        """Test both GPU and CPU configurations use local processing."""
        config = {"strategy": "semantic", "params": {}}
        
        with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf, \
             patch.object(SemanticChunker, '_has_gpu', return_value=gpu_available):
            
            mock_hf.return_value = MagicMock()
            chunker = ChunkingFactory.create_chunker(config)
            
            # Verify appropriate device is used
            call_kwargs = mock_hf.call_args.kwargs
            expected_device = 'cuda' if gpu_available else 'cpu'
            assert call_kwargs['device'] == expected_device
            
            # Both should be local devices
            assert call_kwargs['device'] in ['cpu', 'cuda']
    
    def test_network_timeout_doesnt_leak_data(self):
        """Test that network timeouts don't accidentally leak data."""
        config = {"strategy": "semantic", "params": {}}
        
        # Mock a network timeout scenario
        def timeout_on_network(*args, **kwargs):
            import socket
            raise socket.timeout("Network timeout")
        
        with patch('requests.post', side_effect=timeout_on_network), \
             patch('requests.get', side_effect=timeout_on_network):
            
            # Should use local embeddings without network
            with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
                mock_hf.return_value = MagicMock()
                
                chunker = ChunkingFactory.create_chunker(config)
                
                # Process sensitive content
                chunks = chunker.chunk_text(SENSITIVE_TEST_CONTENT[0], "timeout_test")
                assert len(chunks) >= 1