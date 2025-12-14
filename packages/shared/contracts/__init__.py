"""Shared API contracts for all services."""

from .errors import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    ErrorDetail,
    ErrorResponse,
    InsufficientResourcesErrorResponse,
    NotFoundErrorResponse,
    RateLimitError,
    ServiceUnavailableError,
    ValidationErrorResponse,
    create_insufficient_memory_error,
    create_not_found_error,
    create_validation_error,
)
from .graph import (
    EntityResponse,
    EntitySearchRequest,
    EntitySearchResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    GraphStatsResponse,
    GraphTraversalRequest,
)
from .search import (
    BatchSearchRequest,
    BatchSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    PreloadModelRequest,
    PreloadModelResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    # Search contracts
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "BatchSearchRequest",
    "BatchSearchResponse",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "HybridSearchResult",
    "PreloadModelRequest",
    "PreloadModelResponse",
    # Graph contracts
    "EntityResponse",
    "EntitySearchRequest",
    "EntitySearchResponse",
    "GraphEdge",
    "GraphNode",
    "GraphResponse",
    "GraphStatsResponse",
    "GraphTraversalRequest",
    # Error contracts
    "ErrorResponse",
    "ErrorDetail",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "AuthorizationErrorResponse",
    "NotFoundErrorResponse",
    "InsufficientResourcesErrorResponse",
    "ServiceUnavailableError",
    "RateLimitError",
    "create_validation_error",
    "create_not_found_error",
    "create_insufficient_memory_error",
]
