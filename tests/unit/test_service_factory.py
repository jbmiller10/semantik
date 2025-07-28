#!/usr/bin/env python3
"""
Comprehensive test suite for webui/services/factory.py
Tests service factory functions and dependency injection
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy.ext.asyncio import AsyncSession
from webui.services.collection_service import CollectionService
from webui.services.directory_scan_service import DirectoryScanService
from webui.services.document_scanning_service import DocumentScanningService
from webui.services.factory import (
    create_collection_service,
    create_directory_scan_service,
    create_document_scanning_service,
    create_operation_service,
    create_resource_manager,
    create_search_service,
    get_collection_service,
    get_document_scanning_service,
    get_operation_service,
)
from webui.services.operation_service import OperationService
from webui.services.resource_manager import ResourceManager
from webui.services.search_service import SearchService


class TestServiceFactory:
    """Test service factory functions"""

    @pytest.fixture()
    def mock_db_session(self):
        """Create a mock database session"""
        return AsyncMock(spec=AsyncSession)

    def test_create_collection_service(self, mock_db_session):
        """Test creating CollectionService with dependencies"""
        service = create_collection_service(mock_db_session)

        assert isinstance(service, CollectionService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "collection_repo")
        assert hasattr(service, "operation_repo")
        assert hasattr(service, "document_repo")

    def test_create_document_scanning_service(self, mock_db_session):
        """Test creating DocumentScanningService with dependencies"""
        service = create_document_scanning_service(mock_db_session)

        assert isinstance(service, DocumentScanningService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "document_repo")

    def test_create_operation_service(self, mock_db_session):
        """Test creating OperationService with dependencies"""
        service = create_operation_service(mock_db_session)

        assert isinstance(service, OperationService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "operation_repo")

    @patch("webui.services.factory.httpx.AsyncClient")
    def test_create_search_service(self, mock_httpx_client, mock_db_session):
        """Test creating SearchService with HTTP client"""
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client

        service = create_search_service(mock_db_session)

        assert isinstance(service, SearchService)
        assert service.db_session == mock_db_session
        assert service.http_client == mock_client
        assert hasattr(service, "collection_repo")

    def test_create_directory_scan_service(self, mock_db_session):
        """Test creating DirectoryScanService with dependencies"""
        service = create_directory_scan_service(mock_db_session)

        assert isinstance(service, DirectoryScanService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "document_repo")

    def test_create_resource_manager(self, mock_db_session):
        """Test creating ResourceManager with dependencies"""
        manager = create_resource_manager(mock_db_session)

        assert isinstance(manager, ResourceManager)
        assert hasattr(manager, "collection_repo")
        assert hasattr(manager, "operation_repo")


class TestServiceFactoryDependencyInjection:
    """Test FastAPI dependency injection functions"""

    @pytest.mark.asyncio()
    @patch("webui.services.factory.get_db")
    async def test_get_collection_service_dependency(self, mock_get_db):
        """Test get_collection_service for FastAPI dependency injection"""
        mock_db = AsyncMock(spec=AsyncSession)

        # Mock the async generator
        async def mock_db_generator():
            yield mock_db

        mock_get_db.return_value = mock_db_generator()

        # Test the dependency function
        async for service in get_collection_service():
            assert isinstance(service, CollectionService)
            assert service.db_session == mock_db

    @pytest.mark.asyncio()
    @patch("webui.services.factory.get_db")
    async def test_get_document_scanning_service_dependency(self, mock_get_db):
        """Test get_document_scanning_service for FastAPI dependency injection"""
        mock_db = AsyncMock(spec=AsyncSession)

        async def mock_db_generator():
            yield mock_db

        mock_get_db.return_value = mock_db_generator()

        # Test the dependency function
        async for service in get_document_scanning_service():
            assert isinstance(service, DocumentScanningService)
            assert service.db_session == mock_db

    @pytest.mark.asyncio()
    @patch("webui.services.factory.get_db")
    async def test_get_operation_service_dependency(self, mock_get_db):
        """Test get_operation_service for FastAPI dependency injection"""
        mock_db = AsyncMock(spec=AsyncSession)

        async def mock_db_generator():
            yield mock_db

        mock_get_db.return_value = mock_db_generator()

        # Test the dependency function
        async for service in get_operation_service():
            assert isinstance(service, OperationService)
            assert service.db_session == mock_db


class TestServiceFactoryRepositoryCreation:
    """Test repository creation within factory functions"""

    @patch("webui.services.factory.CollectionRepository")
    @patch("webui.services.factory.OperationRepository")
    @patch("webui.services.factory.DocumentRepository")
    def test_repositories_created_with_session(self, mock_doc_repo_class, mock_op_repo_class, mock_coll_repo_class):
        """Test that repositories are created with the database session"""
        mock_db = AsyncMock(spec=AsyncSession)

        # Mock repository instances
        mock_coll_repo = Mock(spec=CollectionRepository)
        mock_op_repo = Mock(spec=OperationRepository)
        mock_doc_repo = Mock(spec=DocumentRepository)

        mock_coll_repo_class.return_value = mock_coll_repo
        mock_op_repo_class.return_value = mock_op_repo
        mock_doc_repo_class.return_value = mock_doc_repo

        # Create service
        service = create_collection_service(mock_db)

        # Verify repositories were created with session
        mock_coll_repo_class.assert_called_once_with(mock_db)
        mock_op_repo_class.assert_called_once_with(mock_db)
        mock_doc_repo_class.assert_called_once_with(mock_db)

        # Verify service has correct repositories
        assert service.collection_repo == mock_coll_repo
        assert service.operation_repo == mock_op_repo
        assert service.document_repo == mock_doc_repo


class TestServiceFactoryEdgeCases:
    """Test edge cases and error scenarios"""

    def test_create_services_with_none_session(self):
        """Test creating services with None session (should not crash)"""
        # These should not crash but services won't work properly
        service1 = create_collection_service(None)
        assert service1.db_session is None

        service2 = create_document_scanning_service(None)
        assert service2.db_session is None

    @patch("webui.services.factory.httpx.AsyncClient")
    def test_search_service_http_client_creation(self, mock_httpx_client):
        """Test HTTP client creation for SearchService"""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client

        service = create_search_service(mock_db)

        # Verify HTTP client was created with correct parameters
        mock_httpx_client.assert_called_once_with(
            timeout=30.0,
            limits=mock_httpx_client.call_args[1]["limits"],
        )

        # Check limits
        limits = mock_httpx_client.call_args[1]["limits"]
        assert limits.max_keepalive_connections == 5
        assert limits.max_connections == 10


class TestServiceFactoryIntegration:
    """Test service factory integration scenarios"""

    @pytest.mark.asyncio()
    async def test_service_lifecycle_in_request(self):
        """Test service creation and cleanup in a request lifecycle"""
        mock_db = AsyncMock(spec=AsyncSession)

        # Simulate FastAPI request lifecycle
        service = create_collection_service(mock_db)

        # Use the service
        assert service is not None
        assert isinstance(service, CollectionService)

        # Service should be ready to use
        assert service.db_session == mock_db

    def test_multiple_service_creation(self):
        """Test creating multiple services with same session"""
        mock_db = AsyncMock(spec=AsyncSession)

        # Create multiple services
        coll_service = create_collection_service(mock_db)
        doc_service = create_document_scanning_service(mock_db)
        op_service = create_operation_service(mock_db)

        # All should share the same session
        assert coll_service.db_session == mock_db
        assert doc_service.db_session == mock_db
        assert op_service.db_session == mock_db

        # But have different repository instances
        assert coll_service.collection_repo != doc_service.document_repo
