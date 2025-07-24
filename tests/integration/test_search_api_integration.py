"""Integration test for search_api's use of embedding service.

IMPORTANT: Due to architectural constraints where settings are loaded at module
import time, these tests adapt to whether USE_MOCK_EMBEDDINGS is true or false
in the environment. This is a known limitation that will be addressed
in the CORE-003 refactoring.

These tests verify:
- If USE_MOCK_EMBEDDINGS=true: Tests mock the generate_mock_embedding function
- If USE_MOCK_EMBEDDINGS=false: Tests mock the generate_embedding_async function

This ensures the tests pass regardless of environment configuration while still
verifying the correct code paths are executed.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# NOTE: This test verifies that generate_embedding_async is called correctly,
# but due to settings being loaded at module import time, it may use mock embeddings
# depending on the environment configuration


class TestSearchAPIIntegration:
    """Test the integration between search_api and embedding_service."""

    @pytest.fixture(autouse=True)
    def _setup_env(self) -> None:
        """Set up environment variables for the test."""
        # Save original values
        original_values = {}
        env_vars = {
            "QDRANT_HOST": "localhost",
            "QDRANT_PORT": "6333",
            "DEFAULT_COLLECTION": "test_collection",
            # NOTE: USE_MOCK_EMBEDDINGS cannot be set here as settings are loaded at import time
            "DEFAULT_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
            "DEFAULT_QUANTIZATION": "float32",
            "MODEL_UNLOAD_AFTER_SECONDS": "300",
        }

        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        yield

        # Restore original values
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    @patch("packages.vecpipe.search_api.generate_mock_embedding")
    @patch("packages.vecpipe.search_api.generate_embedding_async")
    @patch("packages.vecpipe.search_utils.AsyncQdrantClient")
    @patch("shared.embedding.EmbeddingService")
    @patch("httpx.AsyncClient.get")
    @patch("httpx.AsyncClient.post")
    def test_search_endpoint_uses_embedding_service(
        self,
        mock_post,
        mock_get,
        mock_embedding_service_class,
        mock_qdrant_client_class,
        mock_generate_embedding_async,
        mock_generate_mock_embedding,
    ) -> None:
        """Test that the /search endpoint correctly uses the embedding service."""
        # Import settings to check which mode we're in

        from shared.config import settings

        # Set up mocks
        # Mock the embedding service instance
        mock_embedding_instance = mock_embedding_service_class.return_value
        mock_embedding_instance.mock_mode = False
        mock_embedding_instance.generate_single_embedding.return_value = [0.1] * 1024

        # Set up the appropriate mock based on settings
        if settings.USE_MOCK_EMBEDDINGS:
            # If using mock embeddings, mock generate_mock_embedding
            mock_generate_mock_embedding.return_value = [0.1] * 1024
        else:
            # If not using mock embeddings, mock generate_embedding_async
            async def mock_embedding_async_func(text, model_name, quantization, instruction=None):
                # This simulates the model_manager calling the embedding service
                return mock_embedding_instance.generate_single_embedding(text, model_name, quantization, instruction)

            mock_generate_embedding_async.side_effect = mock_embedding_async_func

        # Mock Qdrant responses
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: {
                "result": {
                    "points_count": 100,
                    "config": {"params": {"vectors": {"size": 1024}}},
                }
            },
        )
        mock_get.return_value.raise_for_status = AsyncMock()

        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: {
                "result": [
                    {
                        "id": "test-id-1",
                        "score": 0.95,
                        "payload": {
                            "path": "/test/file1.txt",
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Test content 1",
                        },
                    },
                    {
                        "id": "test-id-2",
                        "score": 0.90,
                        "payload": {
                            "path": "/test/file2.txt",
                            "chunk_id": "chunk-2",
                            "doc_id": "doc-2",
                            "content": "Test content 2",
                        },
                    },
                ]
            },
        )
        mock_post.return_value.raise_for_status = AsyncMock()

        # Mock Qdrant client for search_utils
        mock_qdrant_instance = mock_qdrant_client_class.return_value

        # Create mock search results
        from unittest.mock import MagicMock

        mock_result_1 = MagicMock()
        mock_result_1.id = "test-id-1"
        mock_result_1.score = 0.95
        mock_result_1.payload = {
            "path": "/test/file1.txt",
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "content": "Test content 1",
        }

        mock_result_2 = MagicMock()
        mock_result_2.id = "test-id-2"
        mock_result_2.score = 0.90
        mock_result_2.payload = {
            "path": "/test/file2.txt",
            "chunk_id": "chunk-2",
            "doc_id": "doc-2",
            "content": "Test content 2",
        }

        mock_qdrant_instance.search = AsyncMock(return_value=[mock_result_1, mock_result_2])

        # Import and create test client
        from packages.vecpipe.search_api import app

        client = TestClient(app)

        # Make search request
        query_text = "test query"
        response = client.post(
            "/search",
            json={
                "query": query_text,
                "k": 5,
                "search_type": "semantic",
            },
        )

        # Assert response is successful
        assert response.status_code == 200
        result = response.json()
        assert result["query"] == query_text
        assert len(result["results"]) == 2
        assert result["results"][0]["score"] == 0.95
        assert result["results"][0]["path"] == "/test/file1.txt"

        # Verify the appropriate function was called based on settings
        if settings.USE_MOCK_EMBEDDINGS:
            # Verify generate_mock_embedding was called
            mock_generate_mock_embedding.assert_called_once()
            call_args = mock_generate_mock_embedding.call_args
            # The function should have been called with the query text
            assert call_args[0][0] == query_text
        else:
            # Verify generate_embedding_async was called
            mock_generate_embedding_async.assert_called_once()
            call_args = mock_generate_embedding_async.call_args
            # The function should have been called with the query text
            assert call_args[0][0] == query_text

            # If the embedding service was actually called,
            # verify it was called with correct parameters
            if mock_embedding_instance.generate_single_embedding.called:
                service_call_args = mock_embedding_instance.generate_single_embedding.call_args
                assert service_call_args[0][0] == query_text  # First positional arg is the text
                assert service_call_args[0][1] == "Qwen/Qwen3-Embedding-0.6B"  # Model name
                assert service_call_args[0][2] == "float32"  # Quantization
                assert (
                    service_call_args[0][3] == "Represent this sentence for searching relevant passages:"
                )  # Instruction

    @patch("packages.vecpipe.search_api.generate_mock_embedding")
    @patch("packages.vecpipe.search_api.generate_embedding_async")
    @patch("packages.vecpipe.search_utils.AsyncQdrantClient")
    @patch("shared.embedding.EmbeddingService")
    @patch("httpx.AsyncClient.get")
    @patch("httpx.AsyncClient.post")
    def test_search_with_custom_model_params(
        self,
        mock_post,
        mock_get,
        mock_embedding_service_class,
        mock_qdrant_client_class,
        mock_generate_embedding_async,
        mock_generate_mock_embedding,
    ) -> None:
        """Test search with custom model name and quantization parameters."""
        # Import settings to check which mode we're in

        from shared.config import settings

        # Set up mocks
        mock_embedding_instance = mock_embedding_service_class.return_value
        mock_embedding_instance.mock_mode = False
        mock_embedding_instance.generate_single_embedding.return_value = [0.2] * 768

        # Set up the appropriate mock based on settings
        if settings.USE_MOCK_EMBEDDINGS:
            # If using mock embeddings, mock generate_mock_embedding
            mock_generate_mock_embedding.return_value = [0.2] * 768
        else:
            # If not using mock embeddings, mock generate_embedding_async
            async def mock_embedding_async_func(text, model_name, quantization, instruction=None):
                # This simulates the model_manager calling the embedding service
                return mock_embedding_instance.generate_single_embedding(text, model_name, quantization, instruction)

            mock_generate_embedding_async.side_effect = mock_embedding_async_func

        # Mock Qdrant responses
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: {
                "result": {
                    "points_count": 50,
                    "config": {"params": {"vectors": {"size": 768}}},
                }
            },
        )
        mock_get.return_value.raise_for_status = AsyncMock()

        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"result": []},  # Empty results
        )
        mock_post.return_value.raise_for_status = AsyncMock()

        # Mock Qdrant client for search_utils to return empty results
        mock_qdrant_instance = mock_qdrant_client_class.return_value
        mock_qdrant_instance.search = AsyncMock(return_value=[])

        # Import and create test client
        from packages.vecpipe.search_api import app

        client = TestClient(app)

        # Make search request with custom parameters
        query_text = "another test query"
        custom_model = "sentence-transformers/all-MiniLM-L6-v2"
        custom_quantization = "int8"

        response = client.post(
            "/search",
            json={
                "query": query_text,
                "k": 10,
                "search_type": "question",
                "model_name": custom_model,
                "quantization": custom_quantization,
            },
        )

        # Assert response is successful
        assert response.status_code == 200
        result = response.json()
        assert result["query"] == query_text
        assert result["num_results"] == 0  # Empty results
        # Note: model_used will be 'mock' if USE_MOCK_EMBEDDINGS is True in the settings
        # We're testing that the embedding service is called with the right parameters,
        # not the response model_used field which depends on settings.USE_MOCK_EMBEDDINGS

        # Verify the appropriate function was called based on settings
        if settings.USE_MOCK_EMBEDDINGS:
            # Verify generate_mock_embedding was called
            mock_generate_mock_embedding.assert_called_once()
            call_args = mock_generate_mock_embedding.call_args
            # The function should have been called with the query text
            assert call_args[0][0] == query_text
        else:
            # Verify generate_embedding_async was called
            mock_generate_embedding_async.assert_called_once()
            call_args = mock_generate_embedding_async.call_args
            # The function should have been called with the query text
            assert call_args[0][0] == query_text

            # If the embedding service was actually called,
            # verify it was called with custom parameters
            if mock_embedding_instance.generate_single_embedding.called:
                service_call_args = mock_embedding_instance.generate_single_embedding.call_args
                assert service_call_args[0][0] == query_text
                assert service_call_args[0][1] == custom_model
                assert service_call_args[0][2] == custom_quantization
                assert (
                    service_call_args[0][3] == "Represent this question for retrieving supporting documents:"
                )  # question instruction
