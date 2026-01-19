"""Unit tests for benchmark metric functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from shared.benchmarks import (
    QueryEvaluator,
    SearchTiming,
    average_precision,
    collapse_chunks_to_documents,
    compute_all_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from shared.benchmarks.exceptions import BenchmarkValidationError

if TYPE_CHECKING:
    from shared.benchmarks.types import RelevanceJudgment, RetrievedChunk


class TestCollapseChunksToDocuments:
    """Tests for collapse_chunks_to_documents function."""

    def test_basic_deduplication(self) -> None:
        """Should deduplicate and preserve first-hit ranking."""
        chunks: list[RetrievedChunk] = [
            {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
            {"doc_id": "b", "chunk_id": "b1", "score": 0.8},
            {"doc_id": "a", "chunk_id": "a2", "score": 0.7},  # duplicate
            {"doc_id": "c", "chunk_id": "c1", "score": 0.6},
        ]
        result = collapse_chunks_to_documents(chunks)
        assert result == ["a", "b", "c"]

    def test_empty_list(self) -> None:
        """Should handle empty input."""
        result = collapse_chunks_to_documents([])
        assert result == []

    def test_no_duplicates(self) -> None:
        """Should handle input with no duplicates."""
        chunks: list[RetrievedChunk] = [
            {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
            {"doc_id": "b", "chunk_id": "b1", "score": 0.8},
        ]
        result = collapse_chunks_to_documents(chunks)
        assert result == ["a", "b"]

    def test_all_same_document(self) -> None:
        """Should return single doc when all chunks from same document."""
        chunks: list[RetrievedChunk] = [
            {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
            {"doc_id": "a", "chunk_id": "a2", "score": 0.8},
            {"doc_id": "a", "chunk_id": "a3", "score": 0.7},
        ]
        result = collapse_chunks_to_documents(chunks)
        assert result == ["a"]


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self) -> None:
        """All retrieved docs are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c", "d"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_precision(self) -> None:
        """No retrieved docs are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_precision(self) -> None:
        """Some retrieved docs are relevant."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        assert precision_at_k(retrieved, relevant, k=5) == 0.6

    def test_k_larger_than_results(self) -> None:
        """K is larger than number of retrieved docs."""
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # Missing results count as non-relevant (denominator stays k)
        assert precision_at_k(retrieved, relevant, k=10) == pytest.approx(0.2)

    def test_k_zero(self) -> None:
        """K=0 should return 0."""
        retrieved = ["a", "b"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=0) == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list should return 0."""
        assert precision_at_k([], {"a", "b"}, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant set - none of retrieved are relevant."""
        retrieved = ["a", "b", "c"]
        assert precision_at_k(retrieved, set(), k=3) == 0.0


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """All relevant docs are in top-k."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        assert recall_at_k(retrieved, relevant, k=4) == 1.0

    def test_zero_recall(self) -> None:
        """No relevant docs in top-k."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self) -> None:
        """Some relevant docs found."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}  # 'f' not retrieved
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)

    def test_empty_relevant(self) -> None:
        """No relevant docs should return 0."""
        retrieved = ["a", "b"]
        assert recall_at_k(retrieved, set(), k=5) == 0.0

    def test_k_zero(self) -> None:
        """K=0 should return 0."""
        assert recall_at_k(["a"], {"a"}, k=0) == 0.0


class TestMeanReciprocalRank:
    """Tests for mean_reciprocal_rank function."""

    def test_first_position(self) -> None:
        """First result is relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self) -> None:
        """Second result is relevant."""
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_position(self) -> None:
        """Third result is first relevant."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_multiple_relevant(self) -> None:
        """Multiple relevant docs - uses first one found."""
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}  # Both relevant, but 'a' found first
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_no_relevant_found(self) -> None:
        """No relevant doc in results."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty results."""
        assert mean_reciprocal_rank([], {"a"}) == 0.0

    def test_empty_relevant(self) -> None:
        """Empty relevant set."""
        assert mean_reciprocal_rank(["a", "b"], set()) == 0.0


class TestNdcgAtK:
    """Tests for ndcg_at_k function."""

    def test_perfect_ranking(self) -> None:
        """Best docs ranked first."""
        # Perfect order: 3, 2, 1
        retrieved = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 1}
        result = ndcg_at_k(retrieved, grades, k=3)
        assert result == pytest.approx(1.0)

    def test_reversed_ranking(self) -> None:
        """Worst docs ranked first."""
        # Reversed order: 1, 2, 3
        retrieved = ["c", "b", "a"]
        grades = {"a": 3, "b": 2, "c": 1}
        result = ndcg_at_k(retrieved, grades, k=3)
        # DCG = 1/log2(2) + 2/log2(3) + 3/log2(4) = 1 + 1.26 + 1.5 = 3.76
        # IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26 + 0.5 = 4.76
        assert result < 1.0
        assert result == pytest.approx(3.76 / 4.76, rel=0.01)

    def test_no_relevant_docs(self) -> None:
        """All relevance grades are 0."""
        retrieved = ["a", "b", "c"]
        grades = {"a": 0, "b": 0, "c": 0}
        assert ndcg_at_k(retrieved, grades, k=3) == 0.0

    def test_empty_grades(self) -> None:
        """Empty relevance grades dict."""
        assert ndcg_at_k(["a", "b"], {}, k=3) == 0.0

    def test_k_zero(self) -> None:
        """K=0 should return 0."""
        assert ndcg_at_k(["a"], {"a": 3}, k=0) == 0.0

    def test_docs_not_in_grades(self) -> None:
        """Retrieved docs not in grades should be treated as 0."""
        retrieved = ["x", "a", "y"]
        grades = {"a": 3}  # x, y not in grades
        result = ndcg_at_k(retrieved, grades, k=3)
        # Only 'a' at position 2 contributes
        # DCG = 0 + 3/log2(3) + 0 = 1.89
        # IDCG = 3/log2(2) = 3.0
        assert result == pytest.approx(1.89 / 3.0, rel=0.01)

    def test_graded_relevance(self) -> None:
        """Test with various relevance grades."""
        retrieved = ["a", "b", "c", "d"]
        grades = {"a": 3, "b": 1, "c": 2, "d": 0}
        result = ndcg_at_k(retrieved, grades, k=4)
        # Not perfectly ordered, so nDCG < 1.0
        assert 0.0 < result < 1.0


class TestAveragePrecision:
    """Tests for average_precision function."""

    def test_perfect_ranking(self) -> None:
        """All relevant docs at the top."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        # P@1=1, P@2=1 -> AP = (1 + 1) / 2 = 1.0
        assert average_precision(retrieved, relevant) == 1.0

    def test_scattered_relevant(self) -> None:
        """Relevant docs scattered in ranking."""
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        # P@1 = 1/1 = 1.0 (a found)
        # P@3 = 2/3 = 0.667 (b found)
        # AP = (1.0 + 0.667) / 2 = 0.833
        assert average_precision(retrieved, relevant) == pytest.approx(0.833, rel=0.01)

    def test_no_relevant_found(self) -> None:
        """No relevant docs in results."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert average_precision(retrieved, relevant) == 0.0

    def test_not_all_relevant_found(self) -> None:
        """Some relevant docs missing from results."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}  # 'b' not retrieved
        # P@1 = 1/1 = 1.0 (a found)
        # AP = 1.0 / 2 = 0.5 (divide by total relevant)
        assert average_precision(retrieved, relevant) == 0.5

    def test_empty_relevant(self) -> None:
        """Empty relevant set."""
        assert average_precision(["a", "b"], set()) == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list."""
        assert average_precision([], {"a"}) == 0.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_default_k_values(self) -> None:
        """Should compute metrics at k=5, 10, 20 by default."""
        retrieved = ["a", "b", "c"]
        grades = {"a": 3, "b": 2}
        results = compute_all_metrics(retrieved, grades)

        # Check we have metrics for k=5, 10, 20
        k_values_in_results = {r.k_value for r in results if r.k_value is not None}
        assert k_values_in_results == {5, 10, 20}

        # Check we have MRR and AP (no k_value)
        no_k_metrics = [r for r in results if r.k_value is None]
        metric_names = {r.name for r in no_k_metrics}
        assert "mrr" in metric_names
        assert "ap" in metric_names

    def test_custom_k_values(self) -> None:
        """Should use custom k values."""
        retrieved = ["a", "b"]
        grades = {"a": 3}
        results = compute_all_metrics(retrieved, grades, k_values=[3, 7])

        k_values_in_results = {r.k_value for r in results if r.k_value is not None}
        assert k_values_in_results == {3, 7}

    def test_all_metric_types(self) -> None:
        """Should compute precision, recall, ndcg, mrr, and ap."""
        retrieved = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 1}
        results = compute_all_metrics(retrieved, grades, k_values=[5])

        metric_names = {r.name for r in results}
        assert metric_names == {"precision", "recall", "ndcg", "mrr", "ap"}

    def test_binary_conversion_from_grades(self) -> None:
        """Grade > 0 should be considered relevant for binary metrics."""
        retrieved = ["a", "b", "c", "d"]
        grades = {"a": 3, "b": 0, "c": 1, "d": 0}  # a, c relevant
        results = compute_all_metrics(retrieved, grades, k_values=[4])

        # Precision@4 = 2/4 = 0.5 (a and c are relevant)
        precision = next(r for r in results if r.name == "precision" and r.k_value == 4)
        assert precision.value == 0.5


class TestQueryEvaluator:
    """Tests for QueryEvaluator class."""

    def test_basic_evaluation(self) -> None:
        """Should evaluate query and return results."""
        evaluator = QueryEvaluator()

        chunks: list[RetrievedChunk] = [
            {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
            {"doc_id": "b", "chunk_id": "b1", "score": 0.8},
        ]
        judgments: list[RelevanceJudgment] = [
            {"doc_id": "a", "relevance_grade": 3},
            {"doc_id": "c", "relevance_grade": 2},  # not retrieved
        ]
        timing = SearchTiming(search_time_ms=50, rerank_time_ms=10)

        result = evaluator.evaluate(
            query_id=1,
            retrieved_chunks=chunks,
            relevance_judgments=judgments,
            k_values=[5],
            timing=timing,
        )

        assert result.query_id == 1
        assert result.retrieved_doc_ids == ["a", "b"]
        assert result.search_time_ms == 50
        assert result.rerank_time_ms == 10
        assert len(result.metrics) > 0

    def test_empty_k_values_raises(self) -> None:
        """Should raise BenchmarkValidationError for empty k_values."""
        evaluator = QueryEvaluator()
        timing = SearchTiming(search_time_ms=50)

        with pytest.raises(BenchmarkValidationError, match="k_values must not be empty"):
            evaluator.evaluate(
                query_id=1,
                retrieved_chunks=[],
                relevance_judgments=[],
                k_values=[],
                timing=timing,
            )

    def test_with_debug_info(self) -> None:
        """Should include debug info when requested."""
        evaluator = QueryEvaluator()

        chunks: list[RetrievedChunk] = [
            {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
            {"doc_id": "a", "chunk_id": "a2", "score": 0.85},
            {"doc_id": "b", "chunk_id": "b1", "score": 0.8},
        ]
        judgments: list[RelevanceJudgment] = [{"doc_id": "a", "relevance_grade": 3}]
        timing = SearchTiming(search_time_ms=50)

        result = evaluator.evaluate(
            query_id=1,
            retrieved_chunks=chunks,
            relevance_judgments=judgments,
            k_values=[5],
            timing=timing,
            include_debug=True,
        )

        assert result.retrieved_debug is not None
        assert result.retrieved_debug["total_chunks"] == 3
        assert result.retrieved_debug["unique_docs"] == 2
        assert result.retrieved_debug["relevant_docs_in_ground_truth"] == 1

    def test_without_debug_info(self) -> None:
        """Should not include debug info by default."""
        evaluator = QueryEvaluator()
        timing = SearchTiming(search_time_ms=50)

        result = evaluator.evaluate(
            query_id=1,
            retrieved_chunks=[],
            relevance_judgments=[],
            k_values=[5],
            timing=timing,
        )

        assert result.retrieved_debug is None
