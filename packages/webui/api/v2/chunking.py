"""
Chunking API v2 endpoints.

This module provides comprehensive RESTful API endpoints for chunking operations
including strategy management, preview operations, collection processing, and analytics.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status

from shared.chunking.infrastructure.exception_translator import exception_translator
from shared.chunking.infrastructure.exceptions import ApplicationError, ValidationError
from shared.database.exceptions import AccessDeniedError

# All exceptions now handled through the infrastructure layer
# Old chunking_exceptions module deleted as we're PRE-RELEASE
from webui.api.v2.chunking_schemas import (
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
    StrategyInfo,
    StrategyMetrics,
    StrategyRecommendation,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig
from webui.dependencies import get_chunking_orchestrator_dependency, get_collection_for_user_safe
from webui.rate_limiter import check_circuit_breaker, create_rate_limit_decorator, rate_limit_dependency
from webui.services.chunking.orchestrator import ChunkingOrchestrator

# ChunkingStrategyRegistry removed - all strategy logic now in service layer
from webui.services.factory import get_collection_service

if TYPE_CHECKING:
    from fastapi.responses import JSONResponse

    from webui.services.collection_service import CollectionService

ChunkingServiceLike: TypeAlias = ChunkingOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/chunking", tags=["chunking-v2"])

# Backwards-compatible alias for tests that patch the original dependency name
get_collection_for_user = get_collection_for_user_safe


async def _resolve_service_payload(payload: Any) -> Any:
    """Normalize service-layer results into API-compatible responses."""

    if inspect.isawaitable(payload):
        payload = await payload

    to_api = getattr(payload, "to_api_model", None)
    if callable(to_api):
        result = to_api()
        if inspect.isawaitable(result):
            result = await result
        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            result = model_dump()
        return result

    return payload


# Note: Exception handlers should be registered at the app level, not router level
# This function can be imported and registered in main.py if needed
async def application_exception_handler(
    _request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    """Global handler for application exceptions with structured error responses."""
    return cast(
        JSONResponse,
        exception_translator.create_error_response(
            exc,
            exc.correlation_id or str(uuid.uuid4()),
        ),
    )


# Strategy Management Endpoints
@router.get(
    "/strategies",
    response_model=list[StrategyInfo],
    summary="List all available chunking strategies",
)
async def list_strategies(
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> list[StrategyInfo]:
    """
    Get a list of all available chunking strategies with their descriptions,
    best use cases, and default configurations.

    Router is now a thin controller - all logic in service!
    """
    try:
        strategies = await service.get_available_strategies()
        return [entry.to_api_model() for entry in strategies]

    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list strategies",
        ) from e


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyInfo,
    summary="Get detailed information about a specific strategy",
)
async def get_strategy_details(
    strategy_id: str,
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> StrategyInfo:
    """
    Get detailed information about a specific chunking strategy including
    configuration options, performance characteristics, and best practices.

    Router is now a thin controller - all logic in service!
    """
    try:
        strategies = await service.get_available_strategies()
        resolved = next((s for s in strategies if s.id == strategy_id), None)
        if not resolved:
            for candidate in strategies:
                if candidate.id.lower() == strategy_id.lower():
                    resolved = candidate
                    break

        if not resolved:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Strategy '{strategy_id}' not found")

        return resolved.to_api_model()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get strategy details",
        ) from e


@router.post(
    "/strategies/recommend",
    response_model=StrategyRecommendation,
    summary="Get strategy recommendation based on file types",
)
async def recommend_strategy(
    file_types: list[str] = Query(..., description="List of file types to analyze"),
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> StrategyRecommendation:
    """
    Get a strategy recommendation based on the provided file types.
    Analyzes the file types and returns the most suitable chunking strategy.

    Router is now a thin controller - all logic in service!
    """
    try:
        recommendation_payload = await service.recommend_strategy(file_type=file_types[0] if file_types else None)
        resolved = await _resolve_service_payload(recommendation_payload)
        if isinstance(resolved, StrategyRecommendation):
            return resolved
        if isinstance(resolved, dict):
            return StrategyRecommendation.model_validate(resolved)
        raise TypeError(f"Unsupported recommendation payload type: {type(resolved)!r}")

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
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.PREVIEW_RATE))],
)
@create_rate_limit_decorator(RateLimitConfig.PREVIEW_RATE)
async def generate_preview(
    request: Request,  # Required for rate limiting
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
    correlation_id: str = Header(None, alias="X-Correlation-ID"),
) -> PreviewResponse:
    """
    Generate a preview of how content would be chunked using a specific strategy.
    Results are cached for 15 minutes to improve performance.

    Rate limited to 10 requests per minute per user.

    Router is now a thin controller - all logic in service!
    """
    payload = await request.json()
    preview_request = PreviewRequest.model_validate(payload)

    # Generate correlation ID if not provided
    if not correlation_id:
        correlation_id = str(uuid.uuid4())

    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        preview_payload = await service.preview_chunks(
            content=preview_request.content,
            document_id=preview_request.document_id,
            strategy=preview_request.strategy.value,
            config=preview_request.config.model_dump() if preview_request.config else None,
            user_id=_current_user["id"],
            use_cache=True,
            max_chunks=preview_request.max_chunks,
        )

        resolved = await _resolve_service_payload(preview_payload)
        if isinstance(resolved, PreviewResponse):
            response = resolved
        elif isinstance(resolved, dict):
            response = PreviewResponse.model_validate(resolved)
        else:
            raise TypeError(f"Unsupported preview payload type: {type(resolved)!r}")

        response.correlation_id = correlation_id
        return response

    except ApplicationError as e:
        # Translate to HTTP exception
        raise exception_translator.translate_application_to_api(e) from e

    except HTTPException:
        # Allow explicit HTTP errors to bubble up (e.g., our validations)
        raise
    except Exception:
        # Unexpected error - log and return generic error
        logger.exception(
            "Unexpected error in preview endpoint",
            extra={"correlation_id": correlation_id},
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "An unexpected error occurred",
                    "code": "INTERNAL_ERROR",
                    "correlation_id": correlation_id,
                }
            },
        ) from None


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare multiple chunking strategies",
    responses={
        429: {"description": "Rate limit exceeded"},
    },
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.COMPARE_RATE))],
)
@create_rate_limit_decorator(RateLimitConfig.COMPARE_RATE)
async def compare_strategies(
    request: Request,  # Required for rate limiting
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> CompareResponse:
    """
    Compare multiple chunking strategies on the same content.
    Provides side-by-side comparison with quality metrics and recommendations.

    Rate limited to 5 requests per minute per user.

    Router is now a thin controller - all logic in service!
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    payload = await request.json()
    compare_request = CompareRequest.model_validate(payload)

    try:
        # Convert strategy enums to strings for service
        strategy_names = [s.value for s in compare_request.strategies]

        # Convert configs if provided
        configs_dict = None
        if compare_request.configs:
            configs_dict = {}
            for strategy, config in compare_request.configs.items():
                # strategy is already a string key from the dict, not an enum
                configs_dict[strategy] = config.model_dump()

        compare_payload = await service.compare_strategies(
            content=compare_request.content or "",
            strategies=strategy_names,
            base_config=None,
            strategy_configs=configs_dict,
            user_id=_current_user.get("id") if _current_user else None,
            max_chunks_per_strategy=compare_request.max_chunks_per_strategy,
        )

        resolved = await _resolve_service_payload(compare_payload)
        if isinstance(resolved, CompareResponse):
            return resolved
        if isinstance(resolved, dict):
            return CompareResponse.model_validate(resolved)
        raise TypeError(f"Unsupported comparison payload type: {type(resolved)!r}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare strategies",
        ) from e


@router.get(
    "/preview/{preview_id}",
    response_model=PreviewResponse,
    summary="Get cached preview results",
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.READ_RATE))],
)
@create_rate_limit_decorator(RateLimitConfig.READ_RATE)
async def get_cached_preview(
    request: Request,  # Required for rate limiting
    preview_id: str,
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> PreviewResponse:
    """
    Retrieve cached preview results by preview ID.
    Preview results are cached for 15 minutes after generation.

    Router is now a thin controller - all logic in service!
    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        preview_payload: Any = await service.get_cached_preview_by_id(preview_id)

        if not preview_payload:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preview not found or expired",
            )

        resolved = await _resolve_service_payload(preview_payload)
        if isinstance(resolved, PreviewResponse):
            return resolved
        if isinstance(resolved, dict):
            return PreviewResponse.model_validate(resolved)
        raise TypeError(f"Unsupported preview payload type: {type(resolved)!r}")

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
    response_model=None,
    response_class=Response,
)
async def clear_preview_cache(
    preview_id: str,
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
) -> None:
    """
    Clear cached preview results for a specific preview ID.
    """
    try:
        await service.clear_preview_cache(preview_id)
    except Exception as e:
        logger.warning(f"Failed to clear preview cache: {e}")
        # Don't raise error for cache clear failures


# Collection Processing
@router.post(
    "/collections/{collection_uuid}/chunk",
    response_model=ChunkingOperationResponse,
    summary="Start chunking operation on collection",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "Invalid configuration"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.PROCESS_RATE))],
)
@create_rate_limit_decorator(RateLimitConfig.PROCESS_RATE)
async def start_chunking_operation(
    request: Request,  # Required for rate limiting
    collection_uuid: str,  # Changed from collection_id to match dependency
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user_safe),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
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

    payload = await request.json()
    chunking_request = ChunkingOperationRequest.model_validate(payload)

    try:
        config_payload = chunking_request.config.model_dump() if chunking_request.config else {}
        try:
            service.validator.validate_strategy(chunking_request.strategy.value)
            if config_payload:
                service.validator.validate_config(chunking_request.strategy.value, config_payload)
        except ValidationError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        # Create operation record
        operation = await collection_service.create_operation(
            collection_id=collection_uuid,
            operation_type="chunking",
            config={
                "strategy": chunking_request.strategy.value,
                "config": chunking_request.config.model_dump() if chunking_request.config else {},
                "document_ids": chunking_request.document_ids,
                "priority": chunking_request.priority,
            },
            user_id=_current_user["id"],
        )

        websocket_channel = f"chunking:{collection_uuid}:{operation['uuid']}"

        # Dispatch background processing via Celery
        try:
            from webui.tasks import celery_app

            celery_app.send_task("webui.tasks.process_collection_operation", args=[operation["uuid"]])
        except Exception as exc:  # pragma: no cover - defensive dispatch
            logger.warning("Failed to enqueue chunking operation %s: %s", operation["uuid"], exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chunking task could not be queued; retry later.",
            ) from exc

        return ChunkingOperationResponse(
            operation_id=operation["uuid"],
            collection_id=collection_uuid,
            status=ChunkingStatus.PENDING,
            strategy=chunking_request.strategy,
            estimated_time_seconds=None,
            queued_position=1,  # Would be calculated from actual queue
            websocket_channel=websocket_channel,
        )

    except HTTPException:
        raise
    except ApplicationError as e:
        # Translate to HTTP exception
        raise exception_translator.translate_application_to_api(e) from e
    except ValidationError as e:
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
    request: Request,
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user_safe),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
    collection_service: CollectionService = Depends(get_collection_service),
) -> ChunkingOperationResponse:
    """
    Update the chunking strategy for a collection.
    Optionally reprocesses existing documents with the new strategy.
    """
    operation_id = str(uuid.uuid4())
    websocket_channel = f"chunking:{collection_id}:{operation_id}"

    payload = await request.json()
    update_request = ChunkingStrategyUpdate.model_validate(payload)

    try:
        # Update collection configuration
        await collection_service.update_collection(
            collection_id=collection_id,
            updates={
                "chunking_strategy": update_request.strategy.value,
                "chunking_config": update_request.config.model_dump() if update_request.config else {},
            },
            user_id=_current_user["id"],
        )

        if update_request.reprocess_existing:
            operation = await collection_service.create_operation(
                collection_id=collection_id,
                operation_type="rechunking",
                config={
                    "strategy": update_request.strategy.value,
                    "config": update_request.config.model_dump() if update_request.config else {},
                },
                user_id=_current_user["id"],
            )

            try:
                from webui.tasks import celery_app

                celery_app.send_task("webui.tasks.process_collection_operation", args=[operation["uuid"]])
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to enqueue rechunking operation %s: %s", operation["uuid"], exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rechunking task could not be queued; retry later.",
                ) from exc

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
    "/collections/{collection_uuid}/chunks",
    response_model=ChunkListResponse,
    summary="Get chunks with pagination",
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.READ_RATE))],
)
@create_rate_limit_decorator(RateLimitConfig.READ_RATE)
async def get_collection_chunks(
    request: Request,  # Required for rate limiting
    collection_uuid: str,  # Changed from collection_id to match dependency # noqa: ARG001
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    document_id: str | None = Query(None, description="Filter by document"),  # noqa: ARG001
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user_safe),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> ChunkListResponse:
    """
    Get paginated list of chunks for a collection.
    Optionally filter by document ID.
    Rate limited to 60 requests per minute per user.

    Check circuit breaker first.
    """
    # Enforce rate limiting safeguards
    check_circuit_breaker(request)

    try:
        chunks_dto = await service.get_collection_chunks(
            collection_uuid,
            page=page,
            page_size=page_size,
            document_id=document_id,
        )
        payload = await _resolve_service_payload(chunks_dto)
        return cast(ChunkListResponse, payload)
    except ApplicationError as e:
        raise exception_translator.translate_application_to_api(e) from e
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
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    collection: dict = Depends(get_collection_for_user_safe),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> ChunkingStats:
    """
    Get detailed chunking statistics and metrics for a collection.

    Router is now a thin controller - all logic in service!
    """
    try:
        stats_dto = await service.get_collection_statistics(
            collection_id=collection_id,
            user_id=_current_user.get("id", 0),
        )
        return cast(ChunkingStats, stats_dto.to_api_model())

    except ApplicationError as e:
        # Translate to HTTP exception
        raise exception_translator.translate_application_to_api(e) from e
    except HTTPException:
        raise
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
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.ANALYTICS_RATE))],
)
@create_rate_limit_decorator(RateLimitConfig.ANALYTICS_RATE)
async def get_global_metrics(
    request: Request,  # Required for rate limiting
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),  # noqa: ARG001
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
) -> GlobalMetrics:
    """
    Get global chunking metrics across all collections for the specified period.
    Rate limited to 30 requests per minute per user.

    """
    # Check circuit breaker first
    check_circuit_breaker(request)

    try:
        metrics_dto = await service.get_global_metrics(
            period_days=period_days,
            user_id=_current_user.get("id") if _current_user else None,
        )
        payload = await _resolve_service_payload(metrics_dto)
        return cast(GlobalMetrics, payload)
    except ApplicationError as e:
        raise exception_translator.translate_application_to_api(e) from e
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
    period_days: int = Query(30, ge=1, le=365, description="Period in days"),
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> list[StrategyMetrics]:
    """
    Get chunking metrics grouped by strategy for the specified period.

    Router is now a thin controller - all logic in service!
    """
    try:
        # Get DTOs from service (method always returns a list)
        metrics_dtos = await service.get_metrics_by_strategy(
            period_days=period_days,
            user_id=_current_user.get("id") if _current_user else None,
        )

        # Convert DTOs to API response models
        return [cast(StrategyMetrics, dto.to_api_model()) for dto in metrics_dtos]

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
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
) -> QualityAnalysis:
    """
    Analyze chunk quality across collections or for a specific collection.

    """
    try:
        quality_dto = await service.get_quality_scores(
            collection_id=collection_id,
            user_id=_current_user.get("id") if _current_user else None,
        )
        payload = await _resolve_service_payload(quality_dto)
        return cast(QualityAnalysis, payload)
    except ApplicationError as e:
        raise exception_translator.translate_application_to_api(e) from e
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
    request: Request,
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> DocumentAnalysisResponse:
    """
    Analyze a document to recommend the best chunking strategy.
    Provides detailed analysis of document structure and complexity.

    """
    payload = await request.json()
    analysis_request = DocumentAnalysisRequest.model_validate(payload)

    try:
        analysis_dto = await service.analyze_document(
            content=analysis_request.content,
            document_id=analysis_request.document_id,
            file_type=analysis_request.file_type,
            user_id=_current_user.get("id") if _current_user else None,
            deep_analysis=analysis_request.deep_analysis,
        )
        payload = await _resolve_service_payload(analysis_dto)
        return cast(DocumentAnalysisResponse, payload)
    except ApplicationError as e:
        raise exception_translator.translate_application_to_api(e) from e
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
    request: Request,
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
) -> SavedConfiguration:
    """
    Save a custom chunking configuration for reuse.
    Configurations are user-specific and can be set as defaults.

    """
    payload = await request.json()
    config_request = CreateConfigurationRequest.model_validate(payload)

    try:
        user_id = _current_user.get("id") if _current_user else None
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user required")

        config_dto = await service.save_configuration(
            name=config_request.name,
            description=config_request.description,
            strategy=config_request.strategy.value,
            config=config_request.config.model_dump(),
            is_default=config_request.is_default,
            tags=config_request.tags,
            user_id=int(user_id),
        )
        payload = await _resolve_service_payload(config_dto)
        return cast(SavedConfiguration, payload)
    except ApplicationError as e:
        raise exception_translator.translate_application_to_api(e) from e
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
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),  # noqa: ARG001
) -> list[SavedConfiguration]:
    """
    List all saved chunking configurations for the current user.
    Can filter by strategy or default status.

    """
    try:
        user_id = _current_user.get("id") if _current_user else None
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user required")

        configs_dto = await service.list_configurations(
            user_id=int(user_id),
            strategy=strategy.value if strategy else None,
            is_default=is_default,
        )
        resolved = []
        for dto in configs_dto:
            payload = await _resolve_service_payload(dto)
            resolved.append(cast(SavedConfiguration, payload))
        return resolved
    except ApplicationError as e:
        raise exception_translator.translate_application_to_api(e) from e
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
    _current_user: dict[str, Any] = Depends(get_current_user),
    service: ChunkingServiceLike = Depends(get_chunking_orchestrator_dependency),
) -> ChunkingProgress:
    """
    Get the current progress of a chunking operation.
    """
    try:
        user_id = _current_user.get("id") if _current_user else None
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authenticated user required")

        progress = await service.get_chunking_progress(operation_id=operation_id, user_id=int(user_id))
        if not progress:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")

        status_value = progress.get("status")
        status_enum = status_value if isinstance(status_value, ChunkingStatus) else ChunkingStatus(status_value)

        return ChunkingProgress(
            operation_id=operation_id,
            status=status_enum,
            progress_percentage=float(progress.get("progress_percentage", 0.0)),
            documents_processed=int(progress.get("documents_processed", 0)),
            total_documents=int(progress.get("total_documents", 0)),
            chunks_created=int(progress.get("chunks_created", 0)),
            current_document=progress.get("current_document"),
            estimated_time_remaining=progress.get("estimated_time_remaining"),
            errors=progress.get("errors", []) or [],
        )

    except HTTPException:
        raise
    except AccessDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get operation progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get operation progress",
        ) from e
