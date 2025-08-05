#!/usr/bin/env python3
"""
Unit tests for chunking_integration_example module.

This module tests the example integration endpoints and services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.webui.api.chunking_exceptions import ChunkingMemoryError, ChunkingValidationError
from packages.webui.api.chunking_integration_example import ChunkingService, router


class TestChunkingIntegrationExample:
    """Tests for chunking integration example endpoints and services."""

    @pytest.fixture()
    def app(self) -> FastAPI:
        """Create a test FastAPI app with the router."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture()
    def client(self, app: FastAPI) -> TestClient:
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture()
    def mock_correlation_id(self) -> str:
        """Mock correlation ID."""
        return "test-correlation-123"

    def test_process_document_success(
        self,
        client: TestClient,
        mock_correlation_id: str,
    ) -> None:
        """Test successful document processing."""
        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id:
            mock_get_id.return_value = mock_correlation_id

            response = client.post(
                "/api/v2/chunking/process",
                params={"document_id": "doc-123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processed"
            assert data["correlation_id"] == mock_correlation_id

    def test_process_document_missing_id(
        self,
        client: TestClient,
        mock_correlation_id: str,
    ) -> None:
        """Test processing with missing document ID."""
        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id:
            mock_get_id.return_value = mock_correlation_id

            response = client.post(
                "/api/v2/chunking/process",
                params={"document_id": ""},
            )

            # The exception will be raised but FastAPI will handle it
            # In a real app with proper exception handlers, this would return 400
            assert response.status_code == 500  # Without proper exception handler setup

    def test_process_document_with_validation_error(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test that validation error is raised correctly."""
        from packages.webui.api.chunking_integration_example import process_document

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id:
            mock_get_id.return_value = mock_correlation_id

            with pytest.raises(ChunkingValidationError) as exc_info:
                import asyncio
                asyncio.run(process_document("", mock_correlation_id))

            assert exc_info.value.detail == "Document ID is required"
            assert exc_info.value.correlation_id == mock_correlation_id
            assert exc_info.value.field_errors == {"document_id": ["This field is required"]}

    @pytest.mark.asyncio
    async def test_chunking_service_process_large_document_success(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test successful document processing in ChunkingService."""
        service = ChunkingService()

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id, \
             patch("packages.webui.api.chunking_integration_example.psutil.virtual_memory") as mock_memory:
            
            mock_get_id.return_value = mock_correlation_id
            # Mock memory usage below threshold
            mock_memory_info = MagicMock()
            mock_memory_info.percent = 50
            mock_memory_info.used = 8 * 1024 * 1024 * 1024  # 8GB
            mock_memory_info.total = 16 * 1024 * 1024 * 1024  # 16GB
            mock_memory.return_value = mock_memory_info

            result = await service.process_large_document("test content")
            assert result == []  # Empty list as per the example

    @pytest.mark.asyncio
    async def test_chunking_service_high_memory_usage(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test ChunkingService raises error when memory usage is high."""
        service = ChunkingService()

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id, \
             patch("packages.webui.api.chunking_integration_example.psutil.virtual_memory") as mock_memory:
            
            mock_get_id.return_value = mock_correlation_id
            # Mock high memory usage
            mock_memory_info = MagicMock()
            mock_memory_info.percent = 95  # Above 90% threshold
            mock_memory_info.used = 15 * 1024 * 1024 * 1024  # 15GB
            mock_memory_info.total = 16 * 1024 * 1024 * 1024  # 16GB
            mock_memory.return_value = mock_memory_info

            with pytest.raises(ChunkingMemoryError) as exc_info:
                await service.process_large_document("test content")

            assert exc_info.value.detail == "Insufficient memory to process document"
            assert exc_info.value.correlation_id == mock_correlation_id
            assert exc_info.value.operation_id == "doc-processing"
            assert exc_info.value.memory_used == mock_memory_info.used
            assert exc_info.value.memory_limit == mock_memory_info.total
            assert "Try processing smaller documents" in exc_info.value.recovery_hint

    @pytest.mark.asyncio
    async def test_chunking_service_memory_threshold(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test ChunkingService at exactly 90% memory threshold."""
        service = ChunkingService()

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id, \
             patch("packages.webui.api.chunking_integration_example.psutil.virtual_memory") as mock_memory:
            
            mock_get_id.return_value = mock_correlation_id
            # Mock exactly at threshold
            mock_memory_info = MagicMock()
            mock_memory_info.percent = 90  # Exactly at threshold
            mock_memory_info.used = 14.4 * 1024 * 1024 * 1024
            mock_memory_info.total = 16 * 1024 * 1024 * 1024
            mock_memory.return_value = mock_memory_info

            # Should succeed at exactly 90%
            result = await service.process_large_document("test content")
            assert result == []

    def test_router_prefix_and_tags(self) -> None:
        """Test that router has correct prefix and tags."""
        assert router.prefix == "/api/v2/chunking"
        assert "chunking" in router.tags

    def test_example_imports(self) -> None:
        """Test that all required imports are available."""
        # This test verifies the imports work correctly
        from packages.webui.api.chunking_exceptions import (
            ChunkingMemoryError,
            ChunkingValidationError,
        )
        from packages.webui.api.chunking_integration_example import (
            ChunkingService,
            process_document,
            router,
        )

        assert ChunkingMemoryError is not None
        assert ChunkingValidationError is not None
        assert ChunkingService is not None
        assert process_document is not None
        assert router is not None

    @pytest.mark.asyncio
    async def test_correlation_id_dependency(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test that correlation ID dependency works correctly."""
        from packages.webui.api.chunking_integration_example import process_document

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id:
            mock_get_id.return_value = mock_correlation_id

            result = await process_document("doc-456", mock_correlation_id)
            
            assert result["correlation_id"] == mock_correlation_id
            assert result["status"] == "processed"

    @pytest.mark.asyncio
    async def test_chunking_service_empty_content(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test ChunkingService with empty content."""
        service = ChunkingService()

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id, \
             patch("packages.webui.api.chunking_integration_example.psutil.virtual_memory") as mock_memory:
            
            mock_get_id.return_value = mock_correlation_id
            # Mock normal memory usage
            mock_memory_info = MagicMock()
            mock_memory_info.percent = 50
            mock_memory_info.used = 8 * 1024 * 1024 * 1024
            mock_memory_info.total = 16 * 1024 * 1024 * 1024
            mock_memory.return_value = mock_memory_info

            result = await service.process_large_document("")
            assert result == []

    @pytest.mark.asyncio
    async def test_chunking_service_large_content(
        self,
        mock_correlation_id: str,
    ) -> None:
        """Test ChunkingService with large content."""
        service = ChunkingService()
        large_content = "x" * 1000000  # 1MB of text

        with patch("packages.webui.api.chunking_integration_example.get_correlation_id") as mock_get_id, \
             patch("packages.webui.api.chunking_integration_example.psutil.virtual_memory") as mock_memory:
            
            mock_get_id.return_value = mock_correlation_id
            # Mock normal memory usage
            mock_memory_info = MagicMock()
            mock_memory_info.percent = 50
            mock_memory_info.used = 8 * 1024 * 1024 * 1024
            mock_memory_info.total = 16 * 1024 * 1024 * 1024
            mock_memory.return_value = mock_memory_info

            result = await service.process_large_document(large_content)
            assert result == []

    def test_exception_details(self) -> None:
        """Test that exceptions have correct details."""
        correlation_id = "test-123"
        
        # Test ChunkingValidationError
        validation_error = ChunkingValidationError(
            detail="Test validation error",
            correlation_id=correlation_id,
            field_errors={"field1": ["error1"]},
        )
        assert validation_error.detail == "Test validation error"
        assert validation_error.correlation_id == correlation_id
        assert validation_error.field_errors == {"field1": ["error1"]}

        # Test ChunkingMemoryError
        memory_error = ChunkingMemoryError(
            detail="Test memory error",
            correlation_id=correlation_id,
            operation_id="op-123",
            memory_used=1000,
            memory_limit=2000,
            recovery_hint="Test hint",
        )
        assert memory_error.detail == "Test memory error"
        assert memory_error.correlation_id == correlation_id
        assert memory_error.operation_id == "op-123"
        assert memory_error.memory_used == 1000
        assert memory_error.memory_limit == 2000
        assert memory_error.recovery_hint == "Test hint"