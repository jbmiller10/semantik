"""Benchmarks module for search quality evaluation.

This module provides tools for evaluating search quality using standard
information retrieval metrics. It includes:

- Core metric functions (precision, recall, MRR, nDCG, AP)
- Query and configuration evaluators
- Type definitions for structured results

Example usage:
    from shared.benchmarks import (
        QueryEvaluator,
        ConfigurationEvaluator,
        compute_all_metrics,
        RetrievedChunk,
        SearchTiming,
    )

    # Evaluate a single query
    evaluator = QueryEvaluator()
    chunks: list[RetrievedChunk] = [
        {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
        {"doc_id": "b", "chunk_id": "b1", "score": 0.8},
    ]
    judgments = [{"doc_id": "a", "relevance_grade": 3}]
    timing = SearchTiming(search_time_ms=50)

    result = evaluator.evaluate(
        query_id=1,
        retrieved_chunks=chunks,
        relevance_judgments=judgments,
        k_values=[5, 10],
        timing=timing,
    )
"""

from .evaluator import ConfigurationEvaluator, QueryEvaluator, SearchFunc
from .exceptions import BenchmarkError, BenchmarkEvaluationError, BenchmarkMetricError, BenchmarkValidationError
from .metrics import (
    average_precision,
    collapse_chunks_to_documents,
    compute_all_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .types import (
    ConfigurationEvaluationResult,
    MetricResult,
    QueryEvaluationResult,
    RelevanceJudgment,
    RetrievedChunk,
    SearchTiming,
)
from .validation import KValueConfig, parse_k_values, validate_top_k_values

__all__ = [
    # Evaluators
    "QueryEvaluator",
    "ConfigurationEvaluator",
    "SearchFunc",
    # Metrics
    "collapse_chunks_to_documents",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "average_precision",
    "compute_all_metrics",
    # Types
    "RetrievedChunk",
    "RelevanceJudgment",
    "MetricResult",
    "SearchTiming",
    "QueryEvaluationResult",
    "ConfigurationEvaluationResult",
    # Exceptions
    "BenchmarkError",
    "BenchmarkMetricError",
    "BenchmarkEvaluationError",
    "BenchmarkValidationError",
    # Validation
    "KValueConfig",
    "parse_k_values",
    "validate_top_k_values",
]
