"""
Data Transfer Objects for chunking application layer.
"""

from .requests import (
    CancelOperationRequest,
    ChunkingStrategy,
    CompareStrategiesRequest,
    GetOperationStatusRequest,
    PreviewRequest,
    ProcessDocumentRequest,
)
from .responses import (
    CancelOperationResponse,
    ChunkDTO,
    CompareStrategiesResponse,
    GetOperationStatusResponse,
    OperationMetrics,
    OperationStatus,
    PreviewResponse,
    ProcessDocumentResponse,
    StrategyMetrics,
)

__all__ = [
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
    "CancelOperationResponse"
]
