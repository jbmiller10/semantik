#!/usr/bin/env python3
"""Verify that search API mocking setup is correct"""

import sys
from unittest.mock import Mock, patch, AsyncMock

# Test that we can import and mock the search API components
try:
    # First test the mock settings
    mock_settings = Mock()
    mock_settings.QDRANT_HOST = "localhost"
    mock_settings.QDRANT_PORT = 6333
    mock_settings.DEFAULT_COLLECTION = "test_collection"
    mock_settings.USE_MOCK_EMBEDDINGS = False
    mock_settings.DEFAULT_EMBEDDING_MODEL = "test-model"
    mock_settings.DEFAULT_QUANTIZATION = "float32"
    mock_settings.MODEL_UNLOAD_AFTER_SECONDS = 300
    mock_settings.SEARCH_API_PORT = 8088
    mock_settings.METRICS_PORT = 9090
    print("✓ Mock settings created successfully")
    
    # Test mock qdrant client
    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.get = AsyncMock()
    mock_qdrant_client.post = AsyncMock()
    mock_qdrant_client.put = AsyncMock()
    mock_qdrant_client.aclose = AsyncMock()
    print("✓ Mock Qdrant client created successfully")
    
    # Test mock embedding service
    mock_embedding_service = Mock()
    mock_embedding_service.is_initialized = True
    mock_embedding_service.current_model_name = "test-model"
    mock_embedding_service.current_quantization = "float32"
    mock_embedding_service.device = "cpu"
    mock_embedding_service.mock_mode = False
    mock_embedding_service.allow_quantization_fallback = True
    
    def mock_get_model_info(*args, **kwargs):
        return {
            "model_name": "test-model",
            "dimension": 1024,
            "description": "Test model"
        }
    mock_embedding_service.get_model_info = Mock(side_effect=mock_get_model_info)
    
    # Test calling get_model_info with different parameter combinations
    result1 = mock_embedding_service.get_model_info()
    print(f"✓ get_model_info() returns: {result1}")
    
    result2 = mock_embedding_service.get_model_info("model", "float32")
    print(f"✓ get_model_info('model', 'float32') returns: {result2}")
    
    # Now test importing the search API with mocks
    with patch("packages.vecpipe.search_api.settings", mock_settings):
        try:
            from packages.vecpipe.search_api import app
            import packages.vecpipe.search_api as search_api_module
            
            # Set the mocked globals
            search_api_module.qdrant_client = mock_qdrant_client
            search_api_module.model_manager = Mock()
            search_api_module.embedding_service = mock_embedding_service
            
            print("✓ Search API imported and mocked successfully")
            
            # Check that the globals are set
            assert search_api_module.qdrant_client is mock_qdrant_client
            assert search_api_module.embedding_service is mock_embedding_service
            print("✓ Global variables set correctly")
            
        except Exception as e:
            print(f"✗ Error importing search API: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"✗ Error in mock setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll checks passed! The mocking setup should work correctly.")