"""Evaluator classes for benchmark query and configuration evaluation.

This module provides the QueryEvaluator for evaluating individual queries
and the ConfigurationEvaluator for running and aggregating results across
multiple queries for a benchmark configuration.
"""

from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from .exceptions import BenchmarkEvaluationError, BenchmarkValidationError
from .metrics import collapse_chunks_to_documents, compute_all_metrics
from .types import (
    ConfigurationEvaluationResult,
    MetricResult,
    QueryEvaluationResult,
    RelevanceJudgment,
    RetrievedChunk,
    SearchTiming,
)


class QueryEvaluator:
    """Evaluates a single query against ground truth relevance judgments.

    This class handles the evaluation of search results for a single query,
    including collapsing chunk-level results to document-level and computing
    all standard IR metrics.

    Example:
        >>> evaluator = QueryEvaluator()
        >>> chunks = [{"doc_id": "a", "chunk_id": "a1", "score": 0.9}]
        >>> judgments = [{"doc_id": "a", "relevance_grade": 3}]
        >>> timing = SearchTiming(search_time_ms=50)
        >>> result = evaluator.evaluate(
        ...     query_id=1,
        ...     retrieved_chunks=chunks,
        ...     relevance_judgments=judgments,
        ...     k_values=[5, 10],
        ...     timing=timing,
        ... )
    """

    def evaluate(
        self,
        query_id: int,
        retrieved_chunks: list[RetrievedChunk],
        relevance_judgments: list[RelevanceJudgment],
        k_values: list[int],
        timing: SearchTiming,
        include_debug: bool = False,
    ) -> QueryEvaluationResult:
        """Evaluate query results against ground truth.

        Args:
            query_id: Database ID of the benchmark query.
            retrieved_chunks: Ordered list of chunks from search results.
            relevance_judgments: Ground truth relevance for documents.
            k_values: List of k values for metrics (e.g., [5, 10, 20]).
            timing: Search and rerank timing information.
            include_debug: Whether to include debug info in results.

        Returns:
            QueryEvaluationResult with computed metrics.

        Raises:
            BenchmarkValidationError: If k_values is empty.
        """
        if not k_values:
            raise BenchmarkValidationError("k_values must not be empty")

        # Collapse chunks to document-level ranking
        retrieved_doc_ids = collapse_chunks_to_documents(retrieved_chunks)

        # Build relevance grades dictionary
        relevance_grades = {judgment["doc_id"]: judgment["relevance_grade"] for judgment in relevance_judgments}

        # Compute all metrics
        metrics = compute_all_metrics(
            retrieved_doc_ids=retrieved_doc_ids,
            relevance_grades=relevance_grades,
            k_values=k_values,
        )

        # Build debug info if requested
        debug_info: dict[str, Any] | None = None
        if include_debug:
            debug_info = {
                "total_chunks": len(retrieved_chunks),
                "unique_docs": len(retrieved_doc_ids),
                "relevant_docs_in_ground_truth": sum(1 for g in relevance_grades.values() if g > 0),
                "top_10_docs": retrieved_doc_ids[:10],
            }

        return QueryEvaluationResult(
            query_id=query_id,
            retrieved_doc_ids=retrieved_doc_ids,
            metrics=metrics,
            search_time_ms=timing.search_time_ms,
            rerank_time_ms=timing.rerank_time_ms,
            retrieved_debug=debug_info,
        )


# Type alias for the search function
SearchFunc = Callable[[str, int], Awaitable[tuple[list[RetrievedChunk], SearchTiming]]]


class ConfigurationEvaluator:
    """Evaluates all queries for a benchmark configuration.

    This class orchestrates the evaluation of multiple queries using a
    search function, aggregating metrics across all queries to produce
    configuration-level results.

    The search function is injected to allow flexibility in how searches
    are performed (different backends, caching, etc.).

    Example:
        >>> async def search(query: str, top_k: int):
        ...     # Perform search and return chunks + timing
        ...     return chunks, SearchTiming(search_time_ms=50)
        ...
        >>> evaluator = ConfigurationEvaluator()
        >>> result = await evaluator.evaluate_configuration(
        ...     config_hash="abc123",
        ...     queries=queries,
        ...     relevance_by_query=relevance_map,
        ...     search_func=search,
        ...     k_values=[5, 10, 20],
        ...     top_k=100,
        ... )
    """

    def __init__(self, query_evaluator: QueryEvaluator | None = None) -> None:
        """Initialize the configuration evaluator.

        Args:
            query_evaluator: QueryEvaluator instance to use. If None,
                a new instance is created.
        """
        self.query_evaluator = query_evaluator or QueryEvaluator()

    async def evaluate_configuration(
        self,
        config_hash: str,
        queries: list[dict[str, Any]],
        relevance_by_query: dict[int, list[RelevanceJudgment]],
        search_func: SearchFunc,
        k_values: list[int],
        top_k: int = 100,
        include_debug: bool = False,
    ) -> ConfigurationEvaluationResult:
        """Run all queries and aggregate metrics.

        Args:
            config_hash: Hash identifying the configuration parameters.
            queries: List of query dicts with 'id' and 'query_text' keys.
            relevance_by_query: Mapping from query_id to relevance judgments.
            search_func: Async function that takes (query_text, top_k) and
                returns (chunks, timing).
            k_values: List of k values for metrics computation.
            top_k: Number of results to retrieve per query.
            include_debug: Whether to include debug info per query.

        Returns:
            ConfigurationEvaluationResult with aggregated metrics.

        Raises:
            BenchmarkValidationError: If inputs are invalid.
            BenchmarkEvaluationError: If evaluation fails.
        """
        if not queries:
            raise BenchmarkValidationError("queries must not be empty")
        if not k_values:
            raise BenchmarkValidationError("k_values must not be empty")

        per_query_results: list[QueryEvaluationResult] = []
        total_search_time_ms = 0
        total_rerank_time_ms = 0

        for query in queries:
            query_id = query["id"]
            query_text = query["query_text"]

            # Get relevance judgments for this query
            relevance_judgments = relevance_by_query.get(query_id, [])

            try:
                # Execute search
                chunks, timing = await search_func(query_text, top_k)

                # Evaluate query
                result = self.query_evaluator.evaluate(
                    query_id=query_id,
                    retrieved_chunks=chunks,
                    relevance_judgments=relevance_judgments,
                    k_values=k_values,
                    timing=timing,
                    include_debug=include_debug,
                )

                per_query_results.append(result)
                total_search_time_ms += result.search_time_ms
                if result.rerank_time_ms is not None:
                    total_rerank_time_ms += result.rerank_time_ms

            except Exception as e:
                raise BenchmarkEvaluationError(
                    f"Search or evaluation failed: {e}",
                    query_id=query_id,
                ) from e

        # Aggregate metrics across all queries
        aggregate_metrics = self._aggregate_metrics(per_query_results)

        return ConfigurationEvaluationResult(
            config_hash=config_hash,
            total_queries=len(per_query_results),
            aggregate_metrics=aggregate_metrics,
            per_query_results=per_query_results,
            total_search_time_ms=total_search_time_ms,
            total_rerank_time_ms=total_rerank_time_ms,
        )

    def _aggregate_metrics(
        self,
        query_results: list[QueryEvaluationResult],
    ) -> list[MetricResult]:
        """Aggregate metrics across all queries by computing mean values.

        Args:
            query_results: List of per-query evaluation results.

        Returns:
            List of MetricResult with mean values across all queries.
        """
        if not query_results:
            return []

        # Group metric values by (name, k_value)
        metric_values: dict[tuple[str, int | None], list[float]] = defaultdict(list)

        for result in query_results:
            for metric in result.metrics:
                key = (metric.name, metric.k_value)
                metric_values[key].append(metric.value)

        # Compute mean for each metric
        aggregate: list[MetricResult] = []
        for (name, k_value), values in sorted(metric_values.items()):
            mean_value = sum(values) / len(values)
            aggregate.append(MetricResult(name=name, k_value=k_value, value=mean_value))

        return aggregate


__all__ = [
    "QueryEvaluator",
    "ConfigurationEvaluator",
    "SearchFunc",
]
