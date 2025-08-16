"""
Service layer DTOs for internal communication.

These DTOs encapsulate business logic and transformations that should not be in API routers.
"""

# API models - the contract between service and API layers
from .api_models import (
    ChunkingConfigBase,
    ChunkingStats,
    ChunkingStrategy,
    ChunkPreview,
    CompareResponse,
    PreviewResponse,
    StrategyComparison,
    StrategyInfo,
    StrategyMetrics,
    StrategyRecommendation,
)

# Service DTOs - internal representations with business logic
from .chunking_dtos import (
    ServiceChunkingStats,
    ServiceChunkPreview,
    ServiceCompareResponse,
    ServicePreviewResponse,
    ServiceStrategyComparison,
    ServiceStrategyInfo,
    ServiceStrategyMetrics,
    ServiceStrategyRecommendation,
)

__all__ = [
    # API models (contract)
    "ChunkingConfigBase",
    "ChunkingStats",
    "ChunkingStrategy",
    "ChunkPreview",
    "CompareResponse",
    "PreviewResponse",
    "StrategyComparison",
    "StrategyInfo",
    "StrategyMetrics",
    "StrategyRecommendation",
    # Service DTOs (internal)
    "ServiceChunkingStats",
    "ServiceChunkPreview",
    "ServiceCompareResponse",
    "ServicePreviewResponse",
    "ServiceStrategyComparison",
    "ServiceStrategyInfo",
    "ServiceStrategyMetrics",
    "ServiceStrategyRecommendation",
]
