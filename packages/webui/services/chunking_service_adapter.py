"""
Adapter service that bridges the old ChunkingService interface to the new architecture.

This is a temporary adapter that allows the existing codebase to work with the
new domain-driven chunking architecture introduced in Phase 1.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.application.dto.requests import (
    ChunkingStrategy,
    CompareStrategiesRequest,
    PreviewRequest,
    ProcessDocumentRequest,
)
from packages.shared.chunking.application.dto.responses import (
    CompareStrategiesResponse,
    PreviewResponse,
    ProcessDocumentResponse,
)
from packages.shared.chunking.application.interfaces.repositories import (
    ChunkingOperationRepository,
    ChunkRepository,
    DocumentRepository,
)
from packages.shared.chunking.application.interfaces.services import (
    ChunkingStrategyFactory,
    DocumentService,
    MetricsService,
    NotificationService,
    UnitOfWork,
)
from packages.shared.chunking.application.use_cases import (
    CompareStrategiesUseCase,
    PreviewChunkingUseCase,
    ProcessDocumentUseCase,
)


class ChunkingServiceAdapter:
    """
    Adapter that provides backward compatibility for the old ChunkingService interface.
    
    This adapter delegates to the new use cases while maintaining the old API contract.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: Any,
        document_repo: Any,
        redis_client: Any,
    ):
        """Initialize the adapter with dependencies."""
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.redis_client = redis_client
        
        # Initialize mock implementations of required services
        # These will be replaced with real implementations in Phase 2
        self._init_services()
        
    def _init_services(self):
        """Initialize service dependencies for use cases."""
        # These are temporary mock implementations
        # Will be replaced with real infrastructure implementations in Phase 2
        self.document_service = MockDocumentService()
        self.strategy_factory = MockStrategyFactory()
        self.notification_service = MockNotificationService()
        self.metrics_service = MockMetricsService()
        self.unit_of_work = MockUnitOfWork(self.db_session)
        
    async def preview_chunking(
        self,
        content: str,
        strategy: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preview chunking results for content.
        
        Delegates to PreviewChunkingUseCase.
        """
        # Map old parameters to new request DTO
        request = PreviewRequest(
            file_path="/tmp/preview.txt",  # Temporary file path
            strategy_type=ChunkingStrategy(strategy),
            min_tokens=kwargs.get("min_tokens", 100),
            max_tokens=kwargs.get("max_tokens", 1000),
            overlap=kwargs.get("overlap", 50),
            preview_size_kb=10,
            max_preview_chunks=5,
        )
        
        # Create and execute use case
        use_case = PreviewChunkingUseCase(
            document_service=self.document_service,
            strategy_factory=self.strategy_factory,
            notification_service=self.notification_service,
            metrics_service=self.metrics_service,
        )
        
        # Store content temporarily for the mock document service
        self.document_service._content = content
        
        response = await use_case.execute(request)
        
        # Map response to old format
        return {
            "preview_id": str(uuid4()),
            "strategy": strategy,
            "config": {"strategy": strategy},
            "chunks": [
                {
                    "content": chunk.content,
                    "position": chunk.position,
                    "token_count": chunk.token_count,
                }
                for chunk in response.preview_chunks
            ],
            "total_chunks": response.estimated_total_chunks,
            "processing_time_ms": response.processing_time_ms,
        }
    
    async def compare_strategies(
        self,
        content: str,
        strategies: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple chunking strategies.
        
        Delegates to CompareStrategiesUseCase.
        """
        request = CompareStrategiesRequest(
            file_path="/tmp/compare.txt",
            strategies=[ChunkingStrategy(s) for s in strategies],
            min_tokens=kwargs.get("min_tokens", 100),
            max_tokens=kwargs.get("max_tokens", 1000),
            overlap=kwargs.get("overlap", 50),
            sample_size_kb=50,
        )
        
        use_case = CompareStrategiesUseCase(
            document_service=self.document_service,
            strategy_factory=self.strategy_factory,
            notification_service=self.notification_service,
            metrics_service=self.metrics_service,
        )
        
        self.document_service._content = content
        
        response = await use_case.execute(request)
        
        return response.to_dict()
    
    async def start_chunking_operation(
        self,
        collection_id: str,
        strategy: str,
        **kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """
        Start a chunking operation for a collection.
        
        Returns WebSocket channel and operation details.
        """
        # Generate operation ID
        operation_id = str(uuid4())
        
        # Create WebSocket channel name
        ws_channel = f"chunking:{operation_id}"
        
        # Return channel and operation details
        operation = {
            "id": operation_id,
            "collection_id": collection_id,
            "strategy": strategy,
            "status": "pending",
        }
        
        return ws_channel, operation
    
    async def validate_config_for_collection(
        self,
        collection_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate chunking configuration for a collection."""
        # Basic validation
        is_valid = True
        errors = []
        
        if not config.get("strategy"):
            is_valid = False
            errors.append("Strategy is required")
            
        return {
            "is_valid": is_valid,
            "errors": errors,
            "estimated_time": 10 if is_valid else 0,
        }
    
    async def track_preview_usage(
        self,
        user_id: str,
        preview_id: str,
        **kwargs
    ) -> None:
        """Track preview usage for rate limiting."""
        # Store in Redis for rate limiting
        if self.redis_client:
            key = f"preview_usage:{user_id}:{preview_id}"
            await self.redis_client.setex(key, 3600, "1")


# Temporary mock implementations
# These will be replaced with real infrastructure implementations in Phase 2

class MockDocumentService:
    """Mock document service for testing."""
    
    def __init__(self):
        self._content = ""
    
    async def load(self, file_path: str, max_size_bytes: Optional[int] = None) -> Any:
        return {"content": self._content, "path": file_path}
    
    async def load_partial(self, file_path: str, size_kb: int) -> Any:
        max_chars = size_kb * 1024
        return {"content": self._content[:max_chars], "path": file_path}
    
    async def extract_text(self, document: Any) -> str:
        return document.get("content", self._content)
    
    async def detect_format(self, file_path: str) -> str:
        return "text"
    
    async def get_metadata(self, file_path: str) -> Dict[str, Any]:
        return {"size": len(self._content), "format": "text"}


class MockStrategyFactory:
    """Mock strategy factory."""
    
    def create_strategy(self, strategy_type: str, config: Dict[str, Any]) -> Any:
        """Create a mock strategy."""
        from packages.shared.chunking.domain.services.chunking_strategies import (
            CharacterBasedStrategy,
        )
        
        # For now, always return character-based strategy
        # This will be expanded in Phase 2
        return CharacterBasedStrategy()
    
    def get_available_strategies(self) -> List[str]:
        return ["character", "recursive", "semantic", "markdown", "hierarchical", "hybrid"]
    
    def get_default_config(self, strategy_type: str) -> Dict[str, Any]:
        return {
            "min_tokens": 100,
            "max_tokens": 1000,
            "overlap": 50,
        }


class MockNotificationService:
    """Mock notification service."""
    
    async def notify_operation_started(self, operation_id: str, metadata: Dict[str, Any]) -> None:
        pass
    
    async def notify_operation_completed(self, operation_id: str, chunks_created: int) -> None:
        pass
    
    async def notify_operation_failed(self, operation_id: str, error: Exception) -> None:
        pass
    
    async def notify_operation_cancelled(self, operation_id: str, reason: Optional[str]) -> None:
        pass
    
    async def notify_progress(self, operation_id: str, progress_percentage: float) -> None:
        pass
    
    async def notify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        pass


class MockMetricsService:
    """Mock metrics service."""
    
    async def record_operation_started(self, operation_id: str) -> None:
        pass
    
    async def record_operation_completed(self, operation_id: str, duration_ms: float) -> None:
        pass
    
    async def record_chunks_created(self, operation_id: str, count: int) -> None:
        pass
    
    async def get_operation_metrics(self, operation_id: str) -> Dict[str, Any]:
        return {}


class MockUnitOfWork:
    """Mock unit of work for transaction management."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.chunking_operations = MockOperationRepository()
        self.chunks = MockChunkRepository()
        self.documents = MockDocumentRepository()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
    
    async def commit(self) -> None:
        pass
    
    async def rollback(self) -> None:
        pass


class MockOperationRepository:
    """Mock operation repository."""
    
    async def create(self, operation: Any) -> str:
        return str(uuid4())
    
    async def find_by_id(self, operation_id: str) -> Any:
        return None
    
    async def update(self, operation: Any) -> None:
        pass
    
    async def delete(self, operation_id: str) -> None:
        pass


class MockChunkRepository:
    """Mock chunk repository."""
    
    async def save(self, chunk: Any) -> str:
        return str(uuid4())
    
    async def save_batch(self, chunks: List[Any]) -> List[str]:
        return [str(uuid4()) for _ in chunks]
    
    async def find_by_id(self, chunk_id: str) -> Any:
        return None
    
    async def find_by_operation(self, operation_id: str) -> List[Any]:
        return []


class MockDocumentRepository:
    """Mock document repository."""
    
    async def find_by_id(self, document_id: str) -> Any:
        return None
    
    async def find_by_path(self, file_path: str) -> Any:
        return None
    
    async def create(self, document: Any) -> str:
        return str(uuid4())