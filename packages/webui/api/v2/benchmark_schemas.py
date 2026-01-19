"""Pydantic schemas for benchmark management endpoints."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Dataset Schemas
# =============================================================================


class DatasetUpload(BaseModel):
    """Request schema for uploading a benchmark dataset."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name for the benchmark dataset",
        json_schema_extra={"example": "MS MARCO Dev Subset"},
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional description of the dataset",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "MS MARCO Dev Subset",
                "description": "A subset of MS MARCO for development testing",
            }
        },
    )


class DatasetResponse(BaseModel):
    """Response schema for benchmark dataset details."""

    id: str = Field(..., description="Unique identifier (UUID)")
    name: str = Field(..., description="Dataset name")
    description: str | None = Field(default=None, description="Dataset description")
    owner_id: int = Field(..., description="ID of the dataset owner")
    query_count: int = Field(..., description="Number of queries in the dataset")
    schema_version: str = Field(..., description="Dataset schema version")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class DatasetListResponse(BaseModel):
    """Response schema for listing datasets."""

    datasets: list[DatasetResponse] = Field(
        ...,
        description="List of benchmark datasets",
    )
    total: int = Field(..., description="Total number of datasets")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Mapping Schemas
# =============================================================================


class MappingCreate(BaseModel):
    """Request schema for creating a dataset-collection mapping."""

    collection_id: str = Field(
        ...,
        description="UUID of the collection to map to",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )

    model_config = ConfigDict(extra="forbid")


class MappingResponse(BaseModel):
    """Response schema for dataset-collection mapping."""

    id: int = Field(..., description="Mapping ID")
    dataset_id: str = Field(..., description="UUID of the benchmark dataset")
    collection_id: str = Field(..., description="UUID of the collection")
    mapping_status: str = Field(..., description="Resolution status: pending, resolved, partial")
    mapped_count: int = Field(..., description="Number of resolved document references")
    total_count: int = Field(..., description="Total number of document references")
    created_at: datetime = Field(..., description="Creation timestamp")
    resolved_at: datetime | None = Field(default=None, description="Resolution timestamp")

    model_config = ConfigDict(from_attributes=True)


class MappingResolveResponse(BaseModel):
    """Response schema for mapping resolution."""

    id: int = Field(..., description="Mapping ID")
    operation_uuid: str | None = Field(
        default=None,
        description="Operation UUID if resolution is running asynchronously (null for synchronous resolution)",
    )
    mapping_status: str = Field(..., description="Resolution status after resolve attempt")
    mapped_count: int = Field(..., description="Number of successfully resolved references")
    total_count: int = Field(..., description="Total number of references")
    unresolved: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of unresolved document references",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Benchmark Configuration Schemas
# =============================================================================


class ConfigMatrixItem(BaseModel):
    """Configuration matrix for benchmark runs.

    Defines the parameter space to explore. Each combination of parameters
    creates a separate benchmark run.
    """

    search_modes: list[Literal["dense", "sparse", "hybrid"]] = Field(
        ...,
        min_length=1,
        description="Search modes to test",
        json_schema_extra={"example": ["dense", "hybrid"]},
    )
    use_reranker: list[bool] = Field(
        ...,
        min_length=1,
        description="Whether to use reranking",
        json_schema_extra={"example": [True, False]},
    )
    top_k_values: list[int] = Field(
        default=[10],
        min_length=1,
        max_length=5,
        description="Top-k values to evaluate at",
        json_schema_extra={"example": [5, 10, 20]},
    )
    rrf_k_values: list[int] = Field(
        default=[60],
        min_length=1,
        description="RRF k constant values for hybrid search",
        json_schema_extra={"example": [60]},
    )
    score_thresholds: list[float | None] = Field(
        default=[None],
        description="Score thresholds (null = no threshold)",
        json_schema_extra={"example": [None, 0.5]},
    )
    primary_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default k value used for UI summaries and per-query stored metrics",
        json_schema_extra={"example": 10},
    )
    k_values_for_metrics: list[int] = Field(
        default=[10],
        min_length=1,
        max_length=10,
        description="All k values to compute metrics for (must include primary_k)",
        json_schema_extra={"example": [5, 10, 20]},
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "search_modes": ["dense", "hybrid"],
                "use_reranker": [True, False],
                "top_k_values": [10],
                "rrf_k_values": [60],
                "score_thresholds": [None],
                "primary_k": 10,
                "k_values_for_metrics": [10],
            }
        },
    )


class BenchmarkCreate(BaseModel):
    """Request schema for creating a benchmark."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name for the benchmark",
        json_schema_extra={"example": "Search Quality Benchmark v1"},
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional description",
    )
    mapping_id: int = Field(
        ...,
        description="ID of the dataset-collection mapping to use",
    )
    config_matrix: ConfigMatrixItem = Field(
        ...,
        description="Configuration matrix defining the parameter space",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default top-k for evaluation",
    )
    metrics_to_compute: list[str] = Field(
        default=["precision", "recall", "mrr", "ndcg"],
        min_length=1,
        description="Metrics to compute",
        json_schema_extra={"example": ["precision", "recall", "mrr", "ndcg"]},
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "Search Quality Benchmark v1",
                "description": "Testing dense vs hybrid search",
                "mapping_id": 1,
                "config_matrix": {
                    "search_modes": ["dense", "hybrid"],
                    "use_reranker": [True],
                    "top_k_values": [10],
                    "rrf_k_values": [60],
                    "score_thresholds": [None],
                },
                "top_k": 10,
                "metrics_to_compute": ["precision", "recall", "mrr", "ndcg"],
            }
        },
    )


# =============================================================================
# Benchmark Response Schemas
# =============================================================================


class BenchmarkResponse(BaseModel):
    """Response schema for benchmark details."""

    id: str = Field(..., description="Unique identifier (UUID)")
    name: str = Field(..., description="Benchmark name")
    description: str | None = Field(default=None, description="Benchmark description")
    owner_id: int = Field(..., description="ID of the benchmark owner")
    mapping_id: int = Field(..., description="ID of the dataset-collection mapping")
    status: str = Field(..., description="Benchmark status: pending, running, completed, failed, cancelled")
    total_runs: int = Field(..., description="Total number of configuration runs")
    completed_runs: int = Field(..., description="Number of completed runs")
    failed_runs: int = Field(..., description="Number of failed runs")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: datetime | None = Field(default=None, description="Start timestamp")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")
    operation_uuid: str | None = Field(default=None, description="UUID of the backing operation")

    model_config = ConfigDict(from_attributes=True)


class BenchmarkListResponse(BaseModel):
    """Response schema for listing benchmarks."""

    benchmarks: list[BenchmarkResponse] = Field(
        ...,
        description="List of benchmarks",
    )
    total: int = Field(..., description="Total number of benchmarks")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")

    model_config = ConfigDict(extra="forbid")


class BenchmarkStartResponse(BaseModel):
    """Response schema for starting a benchmark."""

    id: str = Field(..., description="Benchmark UUID")
    status: str = Field(..., description="New benchmark status")
    operation_uuid: str = Field(..., description="UUID of the created operation")
    message: str = Field(..., description="Status message")

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Benchmark Results Schemas
# =============================================================================


class RunMetricResponse(BaseModel):
    """Response schema for a benchmark run metric."""

    metric_name: str = Field(..., description="Metric name (e.g., precision, recall)")
    k_value: int | None = Field(default=None, description="k value for @k metrics")
    metric_value: float = Field(..., description="Computed metric value")

    model_config = ConfigDict(from_attributes=True)


class RunTimingResponse(BaseModel):
    """Response schema for benchmark run timing."""

    indexing_ms: int | None = Field(default=None, description="Indexing phase duration in ms")
    evaluation_ms: int | None = Field(default=None, description="Evaluation phase duration in ms")
    total_ms: int | None = Field(default=None, description="Total duration in ms")

    model_config = ConfigDict(extra="forbid")


class RunMetricsResponse(BaseModel):
    """Canonical structured metrics for a benchmark run."""

    mrr: float | None = Field(default=None, description="Mean Reciprocal Rank")
    ap: float | None = Field(default=None, description="Average Precision")
    precision: dict[int, float] = Field(default_factory=dict, description="Precision@k keyed by k")
    recall: dict[int, float] = Field(default_factory=dict, description="Recall@k keyed by k")
    ndcg: dict[int, float] = Field(default_factory=dict, description="nDCG@k keyed by k")

    model_config = ConfigDict(extra="forbid")


class BenchmarkRunResponse(BaseModel):
    """Response schema for a benchmark run."""

    id: str = Field(..., description="Run UUID")
    run_order: int = Field(..., description="Order of the run within the benchmark")
    config_hash: str = Field(..., description="Hash of the configuration")
    config: dict[str, Any] = Field(..., description="Run configuration")
    status: str = Field(..., description="Run status")
    error_message: str | None = Field(default=None, description="Error message if failed")
    metrics: RunMetricsResponse = Field(..., description="Canonical structured metrics")
    metrics_flat: dict[str, float] = Field(
        default_factory=dict, description="Compatibility flat metric keys (metric@k)"
    )
    timing: RunTimingResponse = Field(..., description="Timing information")

    model_config = ConfigDict(from_attributes=True)


class BenchmarkResultsResponse(BaseModel):
    """Response schema for benchmark results."""

    benchmark_id: str = Field(..., description="Benchmark UUID")
    primary_k: int = Field(..., description="Default k value used by the UI for summary metrics")
    k_values_for_metrics: list[int] = Field(..., description="All k values computed for metrics")
    runs: list[BenchmarkRunResponse] = Field(..., description="List of benchmark runs with results")
    summary: dict[str, Any] = Field(..., description="Summary statistics")
    total_runs: int = Field(..., description="Total number of runs")

    model_config = ConfigDict(extra="forbid")


class QueryResultResponse(BaseModel):
    """Response schema for per-query results."""

    query_id: int = Field(..., description="Benchmark query ID")
    query_key: str = Field(..., description="Query key from dataset")
    query_text: str = Field(..., description="Query text")
    retrieved_doc_ids: list[str] = Field(..., description="Retrieved document IDs in rank order")
    precision_at_k: float | None = Field(default=None, description="Precision at k")
    recall_at_k: float | None = Field(default=None, description="Recall at k")
    reciprocal_rank: float | None = Field(default=None, description="Reciprocal rank")
    ndcg_at_k: float | None = Field(default=None, description="NDCG at k")
    search_time_ms: int | None = Field(default=None, description="Search time in ms")
    rerank_time_ms: int | None = Field(default=None, description="Rerank time in ms")

    model_config = ConfigDict(from_attributes=True)


class RunQueryResultsResponse(BaseModel):
    """Response schema for paginated query results."""

    run_id: str = Field(..., description="Benchmark run UUID")
    results: list[QueryResultResponse] = Field(..., description="Per-query results")
    total: int = Field(..., description="Total number of query results")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")

    model_config = ConfigDict(extra="forbid")
