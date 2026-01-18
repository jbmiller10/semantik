"""
Benchmarks API v2 endpoints.

This module provides RESTful API endpoints for managing benchmarks,
including creation, execution, and result retrieval.

Error Handling:
    All service-layer exceptions (EntityNotFoundError, AccessDeniedError, etc.)
    are handled by global exception handlers registered in middleware/exception_handlers.py.
    Routers should NOT catch and re-raise these as HTTPExceptions.
"""

import contextlib
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status

from shared.database.models import BenchmarkStatus
from webui.api.schemas import ErrorResponse
from webui.api.v2.benchmark_schemas import (
    BenchmarkCreate,
    BenchmarkListResponse,
    BenchmarkResponse,
    BenchmarkResultsResponse,
    BenchmarkRunResponse,
    BenchmarkStartResponse,
    RunQueryResultsResponse,
    RunTimingResponse,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig
from webui.rate_limiter import limiter
from webui.services.benchmark_service import BenchmarkService
from webui.services.factory import get_benchmark_service

router = APIRouter(prefix="/api/v2/benchmarks", tags=["benchmarks-v2"])


def _benchmark_to_response(benchmark: Any) -> BenchmarkResponse:
    """Convert Benchmark model to response schema."""
    return BenchmarkResponse(
        id=str(benchmark.id),
        name=str(benchmark.name),
        description=str(benchmark.description) if benchmark.description else None,
        owner_id=int(benchmark.owner_id),
        mapping_id=int(benchmark.mapping_id),
        status=str(benchmark.status),
        total_runs=int(benchmark.total_runs),
        completed_runs=int(benchmark.completed_runs),
        failed_runs=int(benchmark.failed_runs),
        created_at=benchmark.created_at,
        started_at=benchmark.started_at,
        completed_at=benchmark.completed_at,
        operation_uuid=str(benchmark.operation_uuid) if benchmark.operation_uuid else None,
    )


@router.post(
    "",
    response_model=BenchmarkResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid configuration"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Mapping not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def create_benchmark(
    request: Request,
    data: BenchmarkCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    """Create a new benchmark.

    Creates a benchmark with a configuration matrix defining the parameter
    space to evaluate. All configuration combinations are pre-computed and
    stored as benchmark runs.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter

    result = await service.create_benchmark(
        user_id=int(current_user["id"]),
        mapping_id=data.mapping_id,
        name=data.name,
        description=data.description,
        config_matrix=data.config_matrix.model_dump(),
        top_k=data.top_k,
        metrics_to_compute=data.metrics_to_compute,
    )

    return BenchmarkResponse(
        id=result["id"],
        name=result["name"],
        description=result["description"],
        owner_id=result["owner_id"],
        mapping_id=result["mapping_id"],
        status=result["status"],
        total_runs=result["total_runs"],
        completed_runs=result["completed_runs"],
        failed_runs=result["failed_runs"],
        created_at=result["created_at"],
    )


@router.get(
    "",
    response_model=BenchmarkListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def list_benchmarks(
    request: Request,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=100),
    status_filter: str | None = Query(default=None),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkListResponse:
    """List benchmarks.

    Returns all benchmarks owned by the current user with pagination.
    Optionally filter by status.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    offset = (page - 1) * per_page

    # Convert string status to enum if provided
    status_enum: BenchmarkStatus | None = None
    if status_filter:
        with contextlib.suppress(ValueError):
            status_enum = BenchmarkStatus(status_filter.lower())

    benchmarks, total = await service.list_benchmarks(
        user_id=int(current_user["id"]),
        offset=offset,
        limit=per_page,
        status_filter=status_enum,
    )

    return BenchmarkListResponse(
        benchmarks=[_benchmark_to_response(b) for b in benchmarks],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/{benchmark_id}",
    response_model=BenchmarkResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Benchmark not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_benchmark(
    request: Request,
    benchmark_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    """Get a benchmark.

    Returns details of a specific benchmark.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    benchmark = await service.get_benchmark(
        benchmark_id=benchmark_id,
        user_id=int(current_user["id"]),
    )

    return _benchmark_to_response(benchmark)


@router.post(
    "/{benchmark_id}/start",
    response_model=BenchmarkStartResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Benchmark not in PENDING status"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Benchmark not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def start_benchmark(
    request: Request,
    benchmark_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkStartResponse:
    """Start benchmark execution.

    Starts the benchmark by creating an operation and dispatching the
    Celery task for async execution.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    result = await service.start_benchmark(
        benchmark_id=benchmark_id,
        user_id=int(current_user["id"]),
    )

    return BenchmarkStartResponse(
        id=result["id"],
        status=result["status"],
        operation_uuid=result["operation_uuid"],
        message=result["message"],
    )


@router.post(
    "/{benchmark_id}/cancel",
    response_model=BenchmarkResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Cannot cancel benchmark in current status"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Benchmark not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def cancel_benchmark(
    request: Request,
    benchmark_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResponse:
    """Cancel a running benchmark.

    Cancels a benchmark that is in PENDING or RUNNING status.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    await service.cancel_benchmark(
        benchmark_id=benchmark_id,
        user_id=int(current_user["id"]),
    )

    # Get updated benchmark for full response
    benchmark = await service.get_benchmark(
        benchmark_id=benchmark_id,
        user_id=int(current_user["id"]),
    )

    return _benchmark_to_response(benchmark)


@router.get(
    "/{benchmark_id}/results",
    response_model=BenchmarkResultsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Benchmark not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_results(
    request: Request,
    benchmark_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkResultsResponse:
    """Get benchmark results.

    Returns aggregated results across all benchmark runs, including
    metrics and timing information.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    results = await service.get_results(
        benchmark_id=benchmark_id,
        user_id=int(current_user["id"]),
    )

    # Format runs with metrics
    formatted_runs = []
    for run_data in results.get("runs", []):
        formatted_runs.append(
            BenchmarkRunResponse(
                id=run_data["run_id"],
                run_order=run_data["run_order"],
                config_hash=run_data["config_hash"],
                config=run_data["config"],
                status=run_data["status"],
                error_message=run_data.get("error_message"),
                metrics=run_data.get("metrics", {}),
                timing=RunTimingResponse(
                    indexing_ms=run_data.get("timing", {}).get("indexing_ms"),
                    evaluation_ms=run_data.get("timing", {}).get("evaluation_ms"),
                    total_ms=run_data.get("timing", {}).get("total_ms"),
                ),
            )
        )

    return BenchmarkResultsResponse(
        benchmark_id=results["benchmark_id"],
        runs=formatted_runs,
        summary=results.get("summary", {}),
        total_runs=len(formatted_runs),
    )


@router.get(
    "/{benchmark_id}/runs/{run_id}/queries",
    response_model=RunQueryResultsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_run_query_results(
    request: Request,
    benchmark_id: str,
    run_id: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=100),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> RunQueryResultsResponse:
    """Get per-query results for a benchmark run.

    Returns detailed per-query results for a specific benchmark run,
    including retrieved documents and metrics.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    _ = benchmark_id  # Run ownership is validated via benchmark

    results = await service.get_run_query_results(
        run_id=run_id,
        user_id=int(current_user["id"]),
        page=page,
        per_page=per_page,
    )

    return RunQueryResultsResponse(
        run_id=results["run_id"],
        results=results["results"],
        total=results["total"],
        page=results["page"],
        per_page=results["per_page"],
    )


@router.delete(
    "/{benchmark_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Benchmark not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def delete_benchmark(
    request: Request,
    benchmark_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkService = Depends(get_benchmark_service),
) -> None:
    """Delete a benchmark.

    Deletes a benchmark and all its runs and results.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    await service.delete_benchmark(
        benchmark_id=benchmark_id,
        user_id=int(current_user["id"]),
    )
