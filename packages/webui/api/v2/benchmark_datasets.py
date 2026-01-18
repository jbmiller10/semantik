"""
Benchmark Datasets API v2 endpoints.

This module provides RESTful API endpoints for managing benchmark datasets
and their mappings to collections.

Error Handling:
    All service-layer exceptions (EntityNotFoundError, AccessDeniedError, etc.)
    are handled by global exception handlers registered in middleware/exception_handlers.py.
    Routers should NOT catch and re-raise these as HTTPExceptions.
"""

from datetime import datetime
from typing import Any, cast

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status

from webui.api.schemas import ErrorResponse
from webui.api.v2.benchmark_schemas import (
    DatasetListResponse,
    DatasetResponse,
    MappingCreate,
    MappingResolveResponse,
    MappingResponse,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig
from webui.rate_limiter import limiter
from webui.services.benchmark_dataset_service import BenchmarkDatasetService
from webui.services.factory import get_benchmark_dataset_service

router = APIRouter(prefix="/api/v2/benchmark-datasets", tags=["benchmark-datasets-v2"])


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def upload_dataset(
    request: Request,
    name: str = Form(..., min_length=1, max_length=255),
    description: str | None = Form(default=None, max_length=2000),
    file: UploadFile = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> DatasetResponse:
    """Upload a benchmark dataset.

    Upload a JSON file containing benchmark queries with their relevance judgments.
    The file should follow the benchmark dataset schema.

    **File Format:**
    ```json
    {
      "schema_version": "1.0",
      "queries": [
        {
          "query_key": "q1",
          "query_text": "example query",
          "relevant_docs": [
            {"doc_ref": {"uri": "/path/to/doc.txt"}, "relevance_grade": 3}
          ]
        }
      ]
    }
    ```

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    file_content = await file.read()

    result = await service.upload_dataset(
        user_id=int(current_user["id"]),
        name=name,
        description=description,
        file_content=file_content,
    )

    return DatasetResponse(
        id=result["id"],
        name=result["name"],
        description=result["description"],
        owner_id=int(current_user["id"]),
        query_count=result["query_count"],
        schema_version=result["schema_version"],
        created_at=result["created_at"],
    )


@router.get(
    "",
    response_model=DatasetListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def list_datasets(
    request: Request,
    page: int = 1,
    per_page: int = 50,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> DatasetListResponse:
    """List benchmark datasets.

    Returns all benchmark datasets owned by the current user with pagination.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    offset = (page - 1) * per_page

    datasets, total = await service.list_datasets(
        user_id=int(current_user["id"]),
        offset=offset,
        limit=per_page,
    )

    return DatasetListResponse(
        datasets=[
            DatasetResponse(
                id=str(d.id),
                name=str(d.name),
                description=str(d.description) if d.description is not None else None,
                owner_id=cast(int, d.owner_id),
                query_count=cast(int, d.query_count),
                schema_version=str(d.schema_version),
                created_at=cast(datetime, d.created_at),
                updated_at=cast(datetime | None, d.updated_at),
            )
            for d in datasets
        ],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_dataset(
    request: Request,
    dataset_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> DatasetResponse:
    """Get a benchmark dataset.

    Returns details of a specific benchmark dataset.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    dataset = await service.get_dataset(
        dataset_id=dataset_id,
        user_id=int(current_user["id"]),
    )

    return DatasetResponse(
        id=str(dataset.id),
        name=str(dataset.name),
        description=str(dataset.description) if dataset.description is not None else None,
        owner_id=cast(int, dataset.owner_id),
        query_count=cast(int, dataset.query_count),
        schema_version=str(dataset.schema_version),
        created_at=cast(datetime, dataset.created_at),
        updated_at=cast(datetime | None, dataset.updated_at),
    )


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def delete_dataset(
    request: Request,
    dataset_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> None:
    """Delete a benchmark dataset.

    Deletes a benchmark dataset and all its mappings.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    await service.delete_dataset(
        dataset_id=dataset_id,
        user_id=int(current_user["id"]),
    )


# =============================================================================
# Mapping Endpoints
# =============================================================================


@router.post(
    "/{dataset_id}/mappings",
    response_model=MappingResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Dataset or collection not found"},
        409: {"model": ErrorResponse, "description": "Mapping already exists"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def create_mapping(
    request: Request,
    dataset_id: str,
    data: MappingCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> MappingResponse:
    """Create a dataset-collection mapping.

    Maps a benchmark dataset to a collection, copying relevance judgments
    that can then be resolved to actual documents.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    result = await service.create_mapping(
        dataset_id=dataset_id,
        collection_id=data.collection_id,
        user_id=int(current_user["id"]),
    )

    return MappingResponse(
        id=result["id"],
        dataset_id=result["dataset_id"],
        collection_id=result["collection_id"],
        mapping_status=result["mapping_status"],
        mapped_count=result["mapped_count"],
        total_count=result["total_count"],
        created_at=result["created_at"],
    )


@router.get(
    "/{dataset_id}/mappings",
    response_model=list[MappingResponse],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def list_mappings(
    request: Request,
    dataset_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> list[MappingResponse]:
    """List mappings for a dataset.

    Returns all collection mappings for a benchmark dataset.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    mappings = await service.list_mappings(
        dataset_id=dataset_id,
        user_id=int(current_user["id"]),
    )

    return [
        MappingResponse(
            id=cast(int, m.id),
            dataset_id=str(m.dataset_id),
            collection_id=str(m.collection_id),
            mapping_status=str(m.mapping_status),
            mapped_count=cast(int, m.mapped_count),
            total_count=cast(int, m.total_count),
            created_at=cast(datetime, m.created_at),
            resolved_at=cast(datetime | None, m.resolved_at),
        )
        for m in mappings
    ]


@router.post(
    "/{dataset_id}/mappings/{mapping_id}/resolve",
    response_model=MappingResolveResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Mapping not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def resolve_mapping(
    request: Request,
    dataset_id: str,
    mapping_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> MappingResolveResponse:
    """Resolve document references in a mapping.

    Attempts to match document references in the dataset's relevance judgments
    to actual documents in the mapped collection. Documents are matched by:
    - URI
    - File path
    - Content hash
    - Document ID

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    _ = dataset_id  # Validated via mapping ownership

    result = await service.resolve_mapping(
        mapping_id=mapping_id,
        user_id=int(current_user["id"]),
    )

    return MappingResolveResponse(
        id=result["id"],
        mapping_status=result["mapping_status"],
        mapped_count=result["mapped_count"],
        total_count=result["total_count"],
        unresolved=result["unresolved"],
    )


@router.get(
    "/{dataset_id}/mappings/{mapping_id}",
    response_model=MappingResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Mapping not found"},
    },
)
@limiter.limit(RateLimitConfig.READ_RATE)
async def get_mapping(
    request: Request,
    dataset_id: str,
    mapping_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: BenchmarkDatasetService = Depends(get_benchmark_dataset_service),
) -> MappingResponse:
    """Get mapping details.

    Returns details of a specific dataset-collection mapping.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    _ = dataset_id  # Validated via mapping ownership

    mapping = await service.get_mapping(
        mapping_id=mapping_id,
        user_id=int(current_user["id"]),
    )

    return MappingResponse(
        id=cast(int, mapping.id),
        dataset_id=str(mapping.dataset_id),
        collection_id=str(mapping.collection_id),
        mapping_status=str(mapping.mapping_status),
        mapped_count=cast(int, mapping.mapped_count),
        total_count=cast(int, mapping.total_count),
        created_at=cast(datetime, mapping.created_at),
        resolved_at=cast(datetime | None, mapping.resolved_at),
    )
