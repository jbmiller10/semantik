"""Type definitions for the benchmarks module."""

from dataclasses import dataclass, field
from typing import Any, TypedDict


class RetrievedChunk(TypedDict):
    """Chunk data returned from search.

    Attributes:
        doc_id: Document ID the chunk belongs to.
        chunk_id: Unique chunk identifier.
        score: Relevance score from search (higher is more relevant).
    """

    doc_id: str
    chunk_id: str
    score: float


class RelevanceJudgment(TypedDict):
    """Ground truth relevance judgment for a document.

    Attributes:
        doc_id: Document ID being judged.
        relevance_grade: Relevance level (0=not relevant, 1=marginally,
            2=relevant, 3=highly relevant).
    """

    doc_id: str
    relevance_grade: int


@dataclass(frozen=True)
class MetricResult:
    """Result of a single metric calculation.

    Attributes:
        name: Metric name (e.g., "precision", "recall", "ndcg").
        k_value: The k value used (e.g., 5, 10, 20), or None for metrics
            like MRR that don't use k.
        value: The calculated metric value.
    """

    name: str
    k_value: int | None
    value: float


@dataclass(frozen=True)
class SearchTiming:
    """Timing information for a search operation.

    Attributes:
        search_time_ms: Time spent on vector search in milliseconds.
        rerank_time_ms: Time spent on reranking, if applicable.
    """

    search_time_ms: int
    rerank_time_ms: int | None = None


@dataclass
class QueryEvaluationResult:
    """Evaluation result for a single query.

    Attributes:
        query_id: Database ID of the benchmark query.
        retrieved_doc_ids: Ordered list of document IDs from search results.
        metrics: List of computed metrics for this query.
        search_time_ms: Time spent on search in milliseconds.
        rerank_time_ms: Time spent on reranking, if applicable.
        retrieved_debug: Optional debug info about retrieved results.
    """

    query_id: int
    retrieved_doc_ids: list[str]
    metrics: list[MetricResult]
    search_time_ms: int
    rerank_time_ms: int | None = None
    retrieved_debug: dict[str, Any] | None = None


@dataclass
class ConfigurationEvaluationResult:
    """Aggregated evaluation results for a benchmark configuration.

    Attributes:
        config_hash: Hash identifying the configuration parameters.
        total_queries: Number of queries evaluated.
        aggregate_metrics: Metrics averaged across all queries.
        per_query_results: Individual results for each query.
        total_search_time_ms: Sum of search times across all queries.
        total_rerank_time_ms: Sum of rerank times, if reranking was used.
    """

    config_hash: str
    total_queries: int
    aggregate_metrics: list[MetricResult]
    per_query_results: list[QueryEvaluationResult] = field(default_factory=list)
    total_search_time_ms: int = 0
    total_rerank_time_ms: int = 0


__all__ = [
    "RetrievedChunk",
    "RelevanceJudgment",
    "MetricResult",
    "SearchTiming",
    "QueryEvaluationResult",
    "ConfigurationEvaluationResult",
]
