"""
Service layer DTOs for internal communication.

These DTOs encapsulate business logic and transformations that should not be in API routers.
"""

# API models - the contract between service and API layers
from .api_models import (
    ChunkingConfigBase,
    ChunkingStats,
    ChunkingStrategy,
    ChunkListResponse,
    ChunkPreview,
    CompareResponse,
    DocumentAnalysisResponse,
    GlobalMetrics,
    PreviewResponse,
    QualityAnalysis,
    QualityLevel,
    SavedConfiguration,
    StrategyComparison,
    StrategyInfo,
    StrategyMetrics,
    StrategyRecommendation,
)

# Service DTOs - internal representations with business logic
from .chunking_dtos import (
    ServiceChunkingStats,
    ServiceChunkList,
    ServiceChunkPreview,
    ServiceChunkRecord,
    ServiceCompareResponse,
    ServiceDocumentAnalysis,
    ServiceGlobalMetrics,
    ServicePreviewResponse,
    ServiceQualityAnalysis,
    ServiceSavedConfiguration,
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
    "ChunkListResponse",
    "DocumentAnalysisResponse",
    "GlobalMetrics",
    "CompareResponse",
    "PreviewResponse",
    "QualityAnalysis",
    "QualityLevel",
    "SavedConfiguration",
    "StrategyComparison",
    "StrategyInfo",
    "StrategyMetrics",
    "StrategyRecommendation",
    # Service DTOs (internal)
    "ServiceChunkingStats",
    "ServiceChunkPreview",
    "ServiceChunkList",
    "ServiceChunkRecord",
    "ServiceCompareResponse",
    "ServiceDocumentAnalysis",
    "ServiceGlobalMetrics",
    "ServicePreviewResponse",
    "ServiceQualityAnalysis",
    "ServiceSavedConfiguration",
    "ServiceStrategyComparison",
    "ServiceStrategyInfo",
    "ServiceStrategyMetrics",
    "ServiceStrategyRecommendation",
]
