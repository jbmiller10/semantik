"""
Chunking API v2 endpoints.

This module provides comprehensive RESTful API endpoints for chunking operations
including strategy management, preview operations, collection processing, and analytics.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status

from packages.webui.api.chunking_exceptions import ChunkingMemoryError, ChunkingTimeoutError, ChunkingValidationError
from packages.webui.api.v2.chunking_schemas import (
    ChunkingConfigBase,
    ChunkingOperationRequest,
    ChunkingOperationResponse,
    ChunkingProgress,
    ChunkingStats,
    ChunkingStatus,
    ChunkingStrategy,
    ChunkingStrategyUpdate,
    ChunkListResponse,
    CompareRequest,
    CompareResponse,
    CreateConfigurationRequest,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    GlobalMetrics,
    PreviewRequest,
    PreviewResponse,
    QualityAnalysis,
    SavedConfiguration,
    StrategyComparison,
    StrategyInfo,
    StrategyMetrics,
    StrategyRecommendation,
)
from packages.webui.auth import get_current_user
from packages.webui.config.rate_limits import RateLimitConfig
from packages.webui.dependencies import get_collection_for_user
from packages.webui.rate_limiter import check_circuit_breaker, limiter
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.chunking_strategies import ChunkingStrategyRegistry
from packages.webui.services.chunking_validation import ChunkingInputValidator
from packages.webui.services.collection_service import CollectionService
from packages.webui.services.factory import get_chunking_service, get_collection_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/chunking", tags=["chunking-v2"])


# Strategy Management Endpoints
@router.get(
    "/strategies",
    response_model=list[StrategyInfo],
    summary="List all available chunking strategies",
)
async def list_strategies(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001  # noqa: ARG001
) -> list[StrategyInfo]:
    """
    Get a list of all available chunking strategies with their descriptions,
    best use cases, and default configurations.
    """
    strategies = []
    for strategy_enum in ChunkingStrategy:
        strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy_enum)

        # Create default config based on strategy
        default_config = ChunkingConfigBase(
            strategy=strategy_enum,
            chunk_size=512,
            chunk_overlap=50,
            preserve_sentences=True,
        )

        strategies.append(
            StrategyInfo(
                id=strategy_enum.value,
                name=strategy_def.get("name", strategy_enum.value),
                description=strategy_def.get("description", ""),
                best_for=strategy_def.get("best_for", []),
                pros=strategy_def.get("pros", []),
                cons=strategy_def.get("cons", []),
                default_config=default_config,
                performance_characteristics=strategy_def.get("performance_characteristics", {}),
            )
        )

    return strategies


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyInfo,
    summary="Get detailed information about a specific strategy",
)
async def get_strategy_details(
    strategy_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001  # noqa: ARG001
) -> StrategyInfo:
    """
    Get detailed information about a specific chunking strategy including
    configuration options, performance characteristics, and best practices.
    """
    try:
        strategy_enum = ChunkingStrategy(strategy_id)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy '{strategy_id}' not found",
        ) from err

    strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy_enum)

    default_config = ChunkingConfigBase(
        strategy=strategy_enum,
        chunk_size=512,
        chunk_overlap=50,
        preserve_sentences=True,
    )

    return StrategyInfo(
        id=strategy_enum.value,
        name=strategy_def.get("name", strategy_enum.value),
        description=strategy_def.get("description", ""),
        best_for=strategy_def.get("best_for", []),
        pros=strategy_def.get("pros", []),
        cons=strategy_def.get("cons", []),
        default_config=default_config,
        performance_characteristics=strategy_def.get("performance_characteristics", {}),
    )


@router.post(
    "/strategies/recommend",
    response_model=StrategyRecommendation,
    summary="Get strategy recommendation based on file types",
)
async def recommend_strategy(
    file_types: list[str] = Query(..., description="List of file types to analyze"),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> StrategyRecommendation:
    """
    Get a strategy recommendation based on the provided file types.
    Analyzes the file types and returns the most suitable chunking strategy.
    """
    try:
        recommendation = await service.recommend_strategy(
            file_types=file_types,
            user_id=current_user["id"],
        )

        return StrategyRecommendation(
            recommended_strategy=recommendation["strategy"],
            confidence=recommendation["confidence"],
            reasoning=recommendation["reasoning"],
            alternative_strategies=recommendation.get("alternatives", []),
            suggested_config=ChunkingConfigBase(
                strategy=recommendation["strategy"],
                chunk_size=recommendation.get("chunk_size", 512),
                chunk_overlap=recommendation.get("chunk_overlap", 50),
                preserve_sentences=True,
            ),
        )
    except Exception as e:
        logger.error(f"Failed to get strategy recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate strategy recommendation",
        ) from e


# Preview Operations
@router.post(
    "/preview",
    response_model=PreviewResponse,
    summary="Generate chunk preview with specific strategy",
    responses={
        429: {"description": "Rate limit exceeded"},
        413: {"description": "Content too large"},
        503: {"description": "Circuit breaker open"},
    },
)
@limiter.limit(RateLimitConfig.PREVIEW_RATE)
async def generate_preview(
    request: Request,  # Required for rate limiting
    preview_request: PreviewRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> PreviewResponse:
    """
    Generate a preview of how content would be chunked using a specific strategy.
    Results are cached for 15 minutes to improve performance.

    Rate limited to 10 requests per minute per user.
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    correlation_id = str(uuid.uuid4())

    # Verify user is authenticated (defense in depth)
    if not current_user or "id" not in current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        # Validate input - must have either document_id or content
        if preview_request.document_id is None and preview_request.content is None:
            raise ChunkingValidationError(
                "Either document_id or content must be provided",
                correlation_id=correlation_id,
                field_errors={"request": ["Missing required input"]},
            )

        # Validate content if provided (but allow empty content)
        if preview_request.content is not None and preview_request.content:
            # Comprehensive input validation (skip for empty content)
            ChunkingInputValidator.validate_content(preview_request.content, correlation_id)

            # Check content size
            content_size = len(preview_request.content.encode("utf-8"))
            if content_size > 10 * 1024 * 1024:  # 10MB limit
                raise ChunkingMemoryError(
                    detail="Content too large for preview (max 10MB)",
                    correlation_id=correlation_id,
                    operation_id="preview",
                    memory_used=content_size,
                    memory_limit=10 * 1024 * 1024,
                )

        # Track usage for rate limiting
        await service.track_preview_usage(
            user_id=current_user["id"],
            strategy=preview_request.strategy,
        )

        # Generate preview
        result = await service.preview_chunking(
            document_id=preview_request.document_id,
            content=preview_request.content,
            strategy=preview_request.strategy,
            config=preview_request.config.model_dump() if preview_request.config else None,
            max_chunks=preview_request.max_chunks,
            include_metrics=preview_request.include_metrics,
            user_id=current_user["id"],
        )

        # Ensure config has strategy field
        config_data = result.get("config", {}).copy() if "config" in result else {}
        if "strategy" not in config_data:
            config_data["strategy"] = result["strategy"]

        return PreviewResponse(
            preview_id=result["preview_id"],
            strategy=result["strategy"],
            config=ChunkingConfigBase(**config_data),
            chunks=result["chunks"],
            total_chunks=result["total_chunks"],
            metrics=result.get("metrics"),
            processing_time_ms=result["processing_time_ms"],
            cached=result.get("cached", False),
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )

    except ChunkingMemoryError:
        raise
    except ChunkingValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except ChunkingTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Preview generation failed: {e}", extra={"correlation_id": correlation_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate preview",
        ) from e


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare multiple chunking strategies",
    responses={
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit(RateLimitConfig.COMPARE_RATE)
async def compare_strategies(
    request: Request,  # Required for rate limiting
    compare_request: CompareRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> CompareResponse:
    """
    Compare multiple chunking strategies on the same content.
    Provides side-by-side comparison with quality metrics and recommendations.

    Rate limited to 5 requests per minute per user.
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    correlation_id = str(uuid.uuid4())

    try:
        comparisons = []
        processing_start = datetime.now(UTC)

        # Process each strategy
        for strategy in compare_request.strategies:
            config = None
            if compare_request.configs and strategy.value in compare_request.configs:
                config = compare_request.configs[strategy.value].model_dump()

            try:
                result = await service.preview_chunking(
                    document_id=compare_request.document_id,
                    content=compare_request.content,
                    strategy=strategy,
                    config=config,
                    max_chunks=compare_request.max_chunks_per_strategy,
                    include_metrics=True,
                    user_id=current_user["id"],
                )

                strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy)

                comparisons.append(
                    StrategyComparison(
                        strategy=strategy,
                        config=ChunkingConfigBase(**result["config"]),
                        sample_chunks=result["chunks"][: compare_request.max_chunks_per_strategy],
                        total_chunks=result["total_chunks"],
                        avg_chunk_size=result["metrics"]["avg_chunk_size"],
                        size_variance=result["metrics"]["size_variance"],
                        quality_score=result["metrics"]["quality_score"],
                        processing_time_ms=result["processing_time_ms"],
                        pros=strategy_def.get("pros", []),
                        cons=strategy_def.get("cons", []),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process strategy {strategy}: {e}")
                continue

        # Generate recommendation based on comparison
        best_strategy = max(comparisons, key=lambda x: x.quality_score)

        recommendation = StrategyRecommendation(
            recommended_strategy=best_strategy.strategy,
            confidence=best_strategy.quality_score,
            reasoning=f"Based on quality score analysis, {best_strategy.strategy.value} provides the best chunking for this content",
            alternative_strategies=[c.strategy for c in comparisons if c.strategy != best_strategy.strategy],
            suggested_config=best_strategy.config,
        )

        processing_time = int((datetime.now(UTC) - processing_start).total_seconds() * 1000)

        return CompareResponse(
            comparison_id=correlation_id,
            comparisons=comparisons,
            recommendation=recommendation,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}", extra={"correlation_id": correlation_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare strategies",
        ) from e


@router.get(
    "/preview/{preview_id}",
    response_model=PreviewResponse,
    summary="Get cached preview results",
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_cached_preview(
    request: Request,  # Required for rate limiting
    preview_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> PreviewResponse:
    """
    Retrieve cached preview results by preview ID.
    Preview results are cached for 15 minutes after generation.
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        # Get cached preview using the key-based method
        cache_key = f"preview:{preview_id}:{current_user['id']}"
        result = await service._get_cached_preview_by_key(cache_key)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preview not found or expired",
            )

        # Map the result to PreviewResponse schema
        return PreviewResponse(
            preview_id=result["preview_id"],
            strategy=result["strategy"],
            config=ChunkingConfigBase(**result.get("config", {})),
            chunks=result["chunks"],
            total_chunks=result["total_chunks"],
            metrics=result.get("metrics"),
            processing_time_ms=result["processing_time_ms"],
            cached=result.get("cached", False),
            expires_at=result.get("expires_at", datetime.now(UTC) + timedelta(minutes=15)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve preview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preview",
        ) from e


@router.delete(
    "/preview/{preview_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear preview cache",
)
async def clear_preview_cache(
    preview_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> None:
    """
    Clear cached preview results for a specific preview ID.
    """
    try:
        await service.clear_preview_cache(preview_id, user_id=current_user["id"])
    except Exception as e:
        logger.warning(f"Failed to clear preview cache: {e}")
        # Don't raise error for cache clear failures


# Collection Processing
@router.post(
    "/collections/{collection_id}/chunk",
    response_model=ChunkingOperationResponse,
    summary="Start chunking operation on collection",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "Invalid configuration"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
)
@limiter.limit(RateLimitConfig.PROCESS_RATE)
async def start_chunking_operation(
    request: Request,  # Required for rate limiting
    collection_id: str,
    chunking_request: ChunkingOperationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
    collection_service: CollectionService = Depends(get_collection_service),
) -> ChunkingOperationResponse:
    """
    Start an asynchronous chunking operation on a collection.
    Returns immediately with an operation ID for tracking progress.

    Progress updates are sent via WebSocket on the returned channel.
    Rate limited to 20 requests per hour per user.
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        # First validate the configuration
        validation_result = await service.validate_config_for_collection(
            collection_id=collection_id,
            strategy=chunking_request.strategy.value,
            config=chunking_request.config.model_dump() if chunking_request.config else None,
            user_id=current_user["id"],
        )

        # Check if configuration is valid
        if not validation_result.get("is_valid", True):
            # Return 400 for invalid configuration
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {validation_result.get('reason', 'Configuration validation failed')}",
            )

        # Create operation record
        operation = await collection_service.create_operation(
            collection_id=collection_id,
            operation_type="chunking",
            config={
                "strategy": chunking_request.strategy.value,
                "config": chunking_request.config.model_dump() if chunking_request.config else {},
                "document_ids": chunking_request.document_ids,
                "priority": chunking_request.priority,
            },
            user_id=current_user["id"],
        )

        # Start chunking operation and get WebSocket channel
        websocket_channel, _ = await service.start_chunking_operation(
            collection_id=collection_id,
            strategy=chunking_request.strategy.value,
            config=chunking_request.config.model_dump() if chunking_request.config else None,
            document_ids=chunking_request.document_ids,
            user_id=current_user["id"],
            operation_data=operation,
        )

        # Queue the chunking task
        background_tasks.add_task(
            process_chunking_operation,
            operation_id=operation["uuid"],
            collection_id=collection_id,
            strategy=chunking_request.strategy,
            config=chunking_request.config,
            document_ids=chunking_request.document_ids,
            user_id=current_user["id"],
            websocket_channel=websocket_channel,
            service=service,
        )

        return ChunkingOperationResponse(
            operation_id=operation["uuid"],
            collection_id=collection_id,
            status=ChunkingStatus.PENDING,
            strategy=chunking_request.strategy,
            estimated_time_seconds=validation_result.get("estimated_time"),
            queued_position=1,  # Would be calculated from actual queue
            websocket_channel=websocket_channel,
        )

    except HTTPException:
        raise
    except ChunkingValidationError as e:
        # Return 400 for validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to start chunking operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start chunking operation",
        ) from e


@router.patch(
    "/collections/{collection_id}/chunking-strategy",
    response_model=ChunkingOperationResponse,
    summary="Update collection chunking strategy",
)
async def update_chunking_strategy(
    collection_id: str,
    update_request: ChunkingStrategyUpdate,
    background_tasks: BackgroundTasks,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
    collection_service: CollectionService = Depends(get_collection_service),
) -> ChunkingOperationResponse:
    """
    Update the chunking strategy for a collection.
    Optionally reprocesses existing documents with the new strategy.
    """
    operation_id = str(uuid.uuid4())
    websocket_channel = f"chunking:{collection_id}:{operation_id}"

    try:
        # Update collection configuration
        await collection_service.update_collection(
            collection_id=collection_id,
            updates={
                "chunking_strategy": update_request.strategy.value,
                "chunking_config": update_request.config.model_dump() if update_request.config else {},
            },
            user_id=current_user["id"],
        )

        if update_request.reprocess_existing:
            # Create reprocessing operation
            operation = await collection_service.create_operation(
                collection_id=collection_id,
                operation_type="rechunking",
                config={
                    "strategy": update_request.strategy.value,
                    "config": update_request.config.model_dump() if update_request.config else {},
                },
                user_id=current_user["id"],
            )

            # Queue reprocessing task
            background_tasks.add_task(
                process_chunking_operation,
                operation_id=operation["uuid"],
                collection_id=collection_id,
                strategy=update_request.strategy,
                config=update_request.config,
                document_ids=None,  # Process all documents
                user_id=current_user["id"],
                websocket_channel=websocket_channel,
                service=service,
            )

            return ChunkingOperationResponse(
                operation_id=operation["uuid"],
                collection_id=collection_id,
                status=ChunkingStatus.PENDING,
                strategy=update_request.strategy,
                websocket_channel=websocket_channel,
            )
        # Just update strategy without reprocessing
        return ChunkingOperationResponse(
            operation_id=operation_id,
            collection_id=collection_id,
            status=ChunkingStatus.COMPLETED,
            strategy=update_request.strategy,
            websocket_channel=websocket_channel,
        )

    except Exception as e:
        logger.error(f"Failed to update chunking strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chunking strategy",
        ) from e


@router.get(
    "/collections/{collection_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get chunks with pagination",
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_collection_chunks(
    request: Request,  # Required for rate limiting
    collection_id: str,  # noqa: ARG001
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    document_id: str | None = Query(None, description="Filter by document"),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> ChunkListResponse:
    """
    Get paginated list of chunks for a collection.
    Optionally filter by document ID.
    Rate limited to 60 requests per minute per user.
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        # This would typically query the chunk storage
        # For now, returning mock data structure
        offset = (page - 1) * page_size

        chunks: list[dict[str, Any]] = []  # Would fetch from database/storage
        total = 0  # Would get total count

        return ChunkListResponse(
            chunks=chunks,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total,
        )

    except Exception as e:
        logger.error(f"Failed to fetch chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chunks",
        ) from e


@router.get(
    "/collections/{collection_id}/chunking-stats",
    response_model=ChunkingStats,
    summary="Get chunking statistics for collection",
)
async def get_chunking_stats(
    collection_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> ChunkingStats:
    """
    Get detailed chunking statistics and metrics for a collection.
    """
    try:
        stats = await service.get_chunking_statistics(
            collection_id=collection_id,
            user_id=current_user["id"],
        )

        return ChunkingStats(
            total_chunks=stats.total_chunks,
            total_documents=stats.total_documents,
            avg_chunk_size=stats.average_chunk_size,
            min_chunk_size=getattr(stats, "min_chunk_size", 0),
            max_chunk_size=getattr(stats, "max_chunk_size", 0),
            size_variance=getattr(stats, "size_variance", 0.0),
            strategy_used=ChunkingStrategy(getattr(stats, "strategy", "fixed_size")),
            last_updated=getattr(stats, "last_updated", datetime.now(UTC)),
            processing_time_seconds=getattr(stats, "processing_time", 0.0),
            quality_metrics=stats.performance_metrics if hasattr(stats, "performance_metrics") else {},
        )

    except Exception as e:
        logger.error(f"Failed to fetch chunking stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chunking statistics",
        ) from e


# Analytics Endpoints
@router.get(
    "/metrics",
    response_model=GlobalMetrics,
    summary="Get global chunking metrics",
)
@limiter.limit(RateLimitConfig.ANALYTICS_RATE)
async def get_global_metrics(
    request: Request,  # Required for rate limiting
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> GlobalMetrics:
    """
    Get global chunking metrics across all collections for the specified period.
    Rate limited to 30 requests per minute per user.
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        # This would aggregate metrics from database
        # For now, returning mock structure
        period_end = datetime.now(UTC)
        period_start = period_end - timedelta(days=period_days)

        return GlobalMetrics(
            total_collections_processed=0,
            total_chunks_created=0,
            total_documents_processed=0,
            avg_chunks_per_document=0.0,
            most_used_strategy=ChunkingStrategy.FIXED_SIZE,
            avg_processing_time=0.0,
            success_rate=0.95,
            period_start=period_start,
            period_end=period_end,
        )

    except Exception as e:
        logger.error(f"Failed to fetch global metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch global metrics",
        ) from e


@router.get(
    "/metrics/by-strategy",
    response_model=list[StrategyMetrics],
    summary="Get metrics grouped by strategy",
)
async def get_metrics_by_strategy(
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> list[StrategyMetrics]:
    """
    Get chunking metrics grouped by strategy for the specified period.
    """
    try:
        metrics = []

        for strategy in ChunkingStrategy:
            # Would fetch actual metrics from database
            metrics.append(
                StrategyMetrics(
                    strategy=strategy,
                    usage_count=0,
                    avg_chunk_size=512,
                    avg_processing_time=1.5,
                    success_rate=0.95,
                    avg_quality_score=0.8,
                    best_for_types=ChunkingStrategyRegistry.get_strategy_definition(strategy).get("best_for", []),
                )
            )

        return metrics

    except Exception as e:
        logger.error(f"Failed to fetch strategy metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch strategy metrics",
        ) from e


@router.get(
    "/quality-scores",
    response_model=QualityAnalysis,
    summary="Get chunk quality analysis",
)
async def get_quality_scores(
    collection_id: str | None = Query(None, description="Specific collection ID"),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> QualityAnalysis:
    """
    Analyze chunk quality across collections or for a specific collection.
    """
    try:
        # Would perform actual quality analysis
        return QualityAnalysis(
            overall_quality="good",
            quality_score=0.75,
            coherence_score=0.8,
            completeness_score=0.7,
            size_consistency=0.75,
            recommendations=[
                "Consider using semantic chunking for better coherence",
                "Adjust chunk size for more consistent results",
            ],
            issues_detected=[
                "Some chunks are too small",
                "Overlapping content detected",
            ],
        )

    except Exception as e:
        logger.error(f"Failed to analyze quality: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze chunk quality",
        ) from e


@router.post(
    "/analyze",
    response_model=DocumentAnalysisResponse,
    summary="Analyze document for strategy recommendation",
)
async def analyze_document(
    analysis_request: DocumentAnalysisRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> DocumentAnalysisResponse:
    """
    Analyze a document to recommend the best chunking strategy.
    Provides detailed analysis of document structure and complexity.
    """
    try:
        # Would perform actual document analysis
        recommendation = await service.recommend_strategy(
            file_types=[analysis_request.file_type] if analysis_request.file_type else [],
            user_id=current_user["id"],
        )

        return DocumentAnalysisResponse(
            document_type=analysis_request.file_type or "unknown",
            content_structure={
                "sections": 5,
                "paragraphs": 20,
                "sentences": 100,
                "words": 1500,
            },
            recommended_strategy=StrategyRecommendation(
                recommended_strategy=recommendation["strategy"],
                confidence=recommendation["confidence"],
                reasoning=recommendation["reasoning"],
                alternative_strategies=recommendation.get("alternatives", []),
                suggested_config=ChunkingConfigBase(
                    strategy=recommendation["strategy"],
                    chunk_size=512,
                    chunk_overlap=50,
                    preserve_sentences=True,
                ),
            ),
            estimated_chunks={
                ChunkingStrategy.FIXED_SIZE: 10,
                ChunkingStrategy.SEMANTIC: 8,
                ChunkingStrategy.RECURSIVE: 12,
            },
            complexity_score=0.6,
            special_considerations=[
                "Document contains tables",
                "Mixed language content detected",
            ],
        )

    except Exception as e:
        logger.error(f"Failed to analyze document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze document",
        ) from e


# Configuration Management
@router.post(
    "/configs",
    response_model=SavedConfiguration,
    summary="Save custom chunking configuration",
    status_code=status.HTTP_201_CREATED,
)
async def save_configuration(
    config_request: CreateConfigurationRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> SavedConfiguration:
    """
    Save a custom chunking configuration for reuse.
    Configurations are user-specific and can be set as defaults.
    """
    try:
        config_id = str(uuid.uuid4())

        # Would save to database
        return SavedConfiguration(
            id=config_id,
            name=config_request.name,
            description=config_request.description,
            strategy=config_request.strategy,
            config=config_request.config,
            created_by=current_user["id"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            usage_count=0,
            is_default=config_request.is_default,
            tags=config_request.tags,
        )

    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save configuration",
        ) from e


@router.get(
    "/configs",
    response_model=list[SavedConfiguration],
    summary="List saved configurations",
)
async def list_configurations(
    strategy: ChunkingStrategy | None = Query(None, description="Filter by strategy"),  # noqa: ARG001
    is_default: bool | None = Query(None, description="Filter default configs"),  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> list[SavedConfiguration]:
    """
    List all saved chunking configurations for the current user.
    Can filter by strategy or default status.
    """
    try:
        # Would fetch from database
        configs: list[SavedConfiguration] = []

        return configs

    except Exception as e:
        logger.error(f"Failed to list configurations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list configurations",
        ) from e


# Progress tracking endpoint
@router.get(
    "/operations/{operation_id}/progress",
    response_model=ChunkingProgress,
    summary="Get chunking operation progress",
)
async def get_operation_progress(
    operation_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingService = Depends(get_chunking_service),  # noqa: ARG001
) -> ChunkingProgress:
    """
    Get the current progress of a chunking operation.
    """
    try:
        progress = await service.get_chunking_progress(
            operation_id=operation_id,
            user_id=current_user["id"],
        )

        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Operation not found",
            )

        return ChunkingProgress(
            operation_id=operation_id,
            status=ChunkingStatus(progress["status"]),
            progress_percentage=progress["progress_percentage"],
            documents_processed=progress["documents_processed"],
            total_documents=progress["total_documents"],
            chunks_created=progress["chunks_created"],
            current_document=progress.get("current_document"),
            estimated_time_remaining=progress.get("estimated_time_remaining"),
            errors=progress.get("errors", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get operation progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get operation progress",
        ) from e


# Background task helper
async def process_chunking_operation(
    operation_id: str,
    collection_id: str,
    strategy: ChunkingStrategy,
    config: ChunkingConfigBase | None,
    document_ids: list[str] | None,
    user_id: int,
    websocket_channel: str,
    service: ChunkingService,
) -> None:
    """
    Background task to process chunking operation.
    Delegates to service layer for actual processing.
    """
    await service.process_chunking_operation(
        operation_id=operation_id,
        collection_id=collection_id,
        strategy=strategy.value,
        config=config.model_dump() if config else None,
        document_ids=document_ids,
        user_id=user_id,
        websocket_channel=websocket_channel,
    )
