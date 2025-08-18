#!/usr/bin/env python3

"""
Comprehensive test suite for webui/services/factory.py
Tests service factory functions and dependency injection
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
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
    create_document_scanning_service,
    create_operation_service,
    create_resource_manager,
    create_search_service,
    get_collection_service,
    get_directory_scan_service,
    get_operation_service,
    get_search_service,
)
from webui.services.operation_service import OperationService
from webui.services.resource_manager import ResourceManager
from webui.services.search_service import SearchService


class TestServiceFactory:
    """Test service factory functions"""

    @pytest.fixture()
    def mock_db_session(self) -> None:
        """Create a mock database session"""
        return AsyncMock(spec=AsyncSession)

    def test_create_collection_service(self, mock_db_session) -> None:
        """Test creating CollectionService with dependencies"""
        service = create_collection_service(mock_db_session)

        assert isinstance(service, CollectionService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "collection_repo")
        assert hasattr(service, "operation_repo")
        assert hasattr(service, "document_repo")

    def test_create_document_scanning_service(self, mock_db_session) -> None:
        """Test creating DocumentScanningService with dependencies"""
        service = create_document_scanning_service(mock_db_session)

        assert isinstance(service, DocumentScanningService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "document_repo")

    def test_create_operation_service(self, mock_db_session) -> None:
        """Test creating OperationService with dependencies"""
        service = create_operation_service(mock_db_session)

        assert isinstance(service, OperationService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "operation_repo")

    def test_create_search_service(self, mock_db_session) -> None:
        """Test creating SearchService with default configuration"""
        service = create_search_service(mock_db_session)

        assert isinstance(service, SearchService)
        assert service.db_session == mock_db_session
        assert hasattr(service, "collection_repo")
        assert hasattr(service, "default_timeout")
        assert hasattr(service, "retry_timeout_multiplier")

    def test_create_search_service_with_custom_timeout(self, mock_db_session) -> None:
        """Test creating SearchService with custom timeout configuration"""
        custom_timeout = httpx.Timeout(30.0)
        custom_multiplier = 2.5

        service = create_search_service(
            mock_db_session,
            default_timeout=custom_timeout,
            retry_timeout_multiplier=custom_multiplier,
        )

        assert isinstance(service, SearchService)
        assert service.default_timeout == custom_timeout
        assert service.retry_timeout_multiplier == custom_multiplier

    def test_create_resource_manager(self, mock_db_session) -> None:
        """Test creating ResourceManager with dependencies"""
        manager = create_resource_manager(mock_db_session)

        assert isinstance(manager, ResourceManager)
        assert hasattr(manager, "collection_repo")
        assert hasattr(manager, "operation_repo")


class TestServiceFactoryDependencyInjection:
    """Test FastAPI dependency injection functions"""

    @pytest.mark.asyncio()
    @patch("webui.services.factory.get_db")
    async def test_get_collection_service_dependency(self, mock_get_db) -> None:
        """Test get_collection_service for FastAPI dependency injection"""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_get_db.return_value = mock_db

        # Test the dependency function
        service = await get_collection_service(mock_db)
        assert isinstance(service, CollectionService)
        assert service.db_session == mock_db

    @pytest.mark.asyncio()
    @patch("webui.services.factory.get_db")
    async def test_get_operation_service_dependency(self, mock_get_db) -> None:
        """Test get_operation_service for FastAPI dependency injection"""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_get_db.return_value = mock_db

        # Test the dependency function
        service = await get_operation_service(mock_db)
        assert isinstance(service, OperationService)
        assert service.db_session == mock_db

    @pytest.mark.asyncio()
    @patch("webui.services.factory.get_db")
    async def test_get_search_service_dependency(self, mock_get_db) -> None:
        """Test get_search_service for FastAPI dependency injection"""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_get_db.return_value = mock_db

        # Test the dependency function
        service = await get_search_service(mock_db)
        assert isinstance(service, SearchService)
        assert service.db_session == mock_db

    @pytest.mark.asyncio()
    async def test_get_directory_scan_service_dependency(self) -> None:
        """Test get_directory_scan_service for FastAPI dependency injection"""
        # DirectoryScanService doesn't require database
        service = await get_directory_scan_service()
        assert isinstance(service, DirectoryScanService)


class TestServiceFactoryRepositoryCreation:
    """Test repository creation within factory functions"""

    @patch("webui.services.factory.CollectionRepository")
    @patch("webui.services.factory.OperationRepository")
    @patch("webui.services.factory.DocumentRepository")
    def test_repositories_created_with_session(
        self, mock_doc_repo_class, mock_op_repo_class, mock_coll_repo_class
    ) -> None:
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

    @patch("webui.services.factory.DocumentRepository")
    def test_document_scanning_service_repository_creation(self, mock_doc_repo_class) -> None:
        """Test repository creation for DocumentScanningService"""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_doc_repo = Mock(spec=DocumentRepository)
        mock_doc_repo_class.return_value = mock_doc_repo

        # Create service
        service = create_document_scanning_service(mock_db)

        # Verify repository was created with session
        mock_doc_repo_class.assert_called_once_with(mock_db)
        assert service.document_repo == mock_doc_repo

    @patch("webui.services.factory.CollectionRepository")
    @patch("webui.services.factory.OperationRepository")
    def test_resource_manager_repository_creation(self, mock_op_repo_class, mock_coll_repo_class) -> None:
        """Test repository creation for ResourceManager"""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_coll_repo = Mock(spec=CollectionRepository)
        mock_op_repo = Mock(spec=OperationRepository)

        mock_coll_repo_class.return_value = mock_coll_repo
        mock_op_repo_class.return_value = mock_op_repo

        # Create manager
        manager = create_resource_manager(mock_db)

        # Verify repositories were created
        mock_coll_repo_class.assert_called_once_with(mock_db)
        mock_op_repo_class.assert_called_once_with(mock_db)

        assert manager.collection_repo == mock_coll_repo
        assert manager.operation_repo == mock_op_repo


class TestServiceFactoryEdgeCases:
    """Test edge cases and error scenarios"""

    def test_create_services_with_none_session(self) -> None:
        """Test creating services with None session (should not crash)"""
        # These should not crash but services won't work properly
        service1 = create_collection_service(None)
        assert service1.db_session is None

        service2 = create_document_scanning_service(None)
        assert service2.db_session is None

        service3 = create_operation_service(None)
        assert service3.db_session is None

    def test_search_service_default_parameters(self) -> None:
        """Test SearchService creation with default parameters"""
        mock_db = AsyncMock(spec=AsyncSession)
        service = create_search_service(mock_db)

        # Should have default values
        # SearchService sets default timeout if None is passed
        assert service.default_timeout.pool == 30.0
        assert service.default_timeout.connect == 5.0
        assert service.default_timeout.read == 30.0
        assert service.default_timeout.write == 5.0
        assert service.retry_timeout_multiplier == 4.0


class TestServiceFactoryIntegration:
    """Test factory functions with real instances"""

    @pytest.mark.asyncio()
    async def test_fastapi_dependency_injection_pattern(self) -> None:
        """Test how factory functions integrate with FastAPI dependencies"""
        # Test the direct factory functions (no async dependencies)
        mock_db = AsyncMock()

        # Test collection service
        collection_service = create_collection_service(mock_db)
        assert isinstance(collection_service, CollectionService)

        # Test document scanning service
        doc_scan_service = create_document_scanning_service(mock_db)
        assert isinstance(doc_scan_service, DocumentScanningService)

        # Test operation service
        operation_service = create_operation_service(mock_db)
        assert isinstance(operation_service, OperationService)

        # Test search service
        search_service = create_search_service(mock_db)
        assert isinstance(search_service, SearchService)

        # Test resource manager
        resource_manager = create_resource_manager(mock_db)
        assert isinstance(resource_manager, ResourceManager)

        # Test directory scan service (no db dependency)
        dir_scan_service = await get_directory_scan_service()
        assert isinstance(dir_scan_service, DirectoryScanService)

    def test_multiple_service_creation(self) -> None:
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

    @pytest.mark.asyncio()
    async def test_service_lifecycle_in_request(self) -> None:
        """Test service creation and cleanup in a request lifecycle"""
        mock_db = AsyncMock(spec=AsyncSession)

        # Simulate FastAPI request lifecycle
        service = create_collection_service(mock_db)

        # Use the service
        assert service is not None
        assert isinstance(service, CollectionService)

        # Service should be ready to use
        assert service.db_session == mock_db
