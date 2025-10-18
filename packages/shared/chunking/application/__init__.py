"""
Application layer for chunking system.

This layer orchestrates business operations by coordinating between
the domain layer and infrastructure services. It defines use cases,
DTOs, and interface contracts.

Structure:
- use_cases/: Business operation orchestrators
- dto/: Data Transfer Objects for input/output
- interfaces/: Contracts for repositories and services
"""

# Import use cases
# Import DTOs
from .dto import (  # Request DTOs; Response DTOs
    CancelOperationRequest,
    CancelOperationResponse,
    ChunkDTO,
    ChunkingStrategy,
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    GetOperationStatusRequest,
    GetOperationStatusResponse,
    OperationMetrics,
    OperationStatus,
    PreviewRequest,
    PreviewResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    StrategyMetrics,
)

# Import interfaces
from .interfaces import (  # Repository interfaces; Service interfaces
    CheckpointRepository,
    ChunkingOperationRepository,
    ChunkingStrategyFactory,
    ChunkRepository,
    DocumentFormat,
    DocumentRepository,
    DocumentService,
    MetricsService,
    NotificationService,
    UnitOfWork,
)
from .use_cases import (
    CancelOperationUseCase,
    CompareStrategiesUseCase,
    GetOperationStatusUseCase,
    PreviewChunkingUseCase,
    ProcessDocumentUseCase,
)

__all__ = [
    # Use Cases
    "PreviewChunkingUseCase",
    "ProcessDocumentUseCase",
    "CompareStrategiesUseCase",
    "GetOperationStatusUseCase",
    "CancelOperationUseCase",
    # Request DTOs
    "ChunkingStrategy",
    "PreviewRequest",
    "ProcessDocumentRequest",
    "CompareStrategiesRequest",
    "GetOperationStatusRequest",
    "CancelOperationRequest",
    # Response DTOs
    "OperationStatus",
    "ChunkDTO",
    "PreviewResponse",
    "ProcessDocumentResponse",
    "StrategyMetrics",
    "CompareStrategiesResponse",
    "OperationMetrics",
    "GetOperationStatusResponse",
    "CancelOperationResponse",
    # Repository Interfaces
    "ChunkRepository",
    "ChunkingOperationRepository",
    "CheckpointRepository",
    "DocumentRepository",
    # Service Interfaces
    "DocumentFormat",
    "DocumentService",
    "NotificationService",
    "ChunkingStrategyFactory",
    "MetricsService",
    "UnitOfWork",
]
