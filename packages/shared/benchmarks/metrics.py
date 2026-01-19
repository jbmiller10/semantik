"""Core metric functions for benchmark evaluation.

This module provides standard information retrieval metrics for evaluating
search quality: precision, recall, MRR, nDCG, and average precision.
"""

import math

from .types import MetricResult, RetrievedChunk


def collapse_chunks_to_documents(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """Collapse chunk results to a ranked document list using first-hit deduplication.

    When multiple chunks from the same document appear in search results,
    we keep only the first occurrence (highest-ranked) and remove duplicates.
    This preserves the ranking order based on the best chunk per document.

    Args:
        retrieved_chunks: Ordered list of chunks from search, highest score first.

    Returns:
        Ordered list of unique document IDs, preserving first-hit ranking.

    Example:
        >>> chunks = [
        ...     {"doc_id": "a", "chunk_id": "a1", "score": 0.9},
        ...     {"doc_id": "b", "chunk_id": "b1", "score": 0.8},
        ...     {"doc_id": "a", "chunk_id": "a2", "score": 0.7},  # duplicate
        ... ]
        >>> collapse_chunks_to_documents(chunks)
        ['a', 'b']
    """
    seen: set[str] = set()
    result: list[str] = []
    for chunk in retrieved_chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K: fraction of top-k slots that are relevant.

    Precision@K measures how many of the top-k ranked positions are filled by
    relevant documents. If fewer than k documents are retrieved, the missing
    slots are treated as non-relevant (denominator stays k).

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of document IDs that are relevant (ground truth).
        k: Number of top results to consider.

    Returns:
        Precision value between 0.0 and 1.0.

    Example:
        >>> precision_at_k(['a', 'b', 'c', 'd', 'e'], {'a', 'c', 'e'}, k=5)
        0.6  # 3 relevant out of 5 retrieved
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_count / k


def recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K: fraction of relevant documents found in top-k.

    Recall@K measures how many of the relevant documents were retrieved.
    It answers: "Of all the useful documents, how many did I find?"

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of document IDs that are relevant (ground truth).
        k: Number of top results to consider.

    Returns:
        Recall value between 0.0 and 1.0. Returns 0.0 if there are no
        relevant documents in the ground truth.

    Example:
        >>> recall_at_k(['a', 'b', 'c', 'd', 'e'], {'a', 'c', 'f'}, k=5)
        0.667  # 2 out of 3 relevant docs found
    """
    if k <= 0 or not relevant_doc_ids:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_count / len(relevant_doc_ids)


def mean_reciprocal_rank(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR): 1/rank of first relevant document.

    MRR measures how early the first relevant result appears. It's particularly
    useful when users care most about finding at least one good result quickly.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of document IDs that are relevant (ground truth).

    Returns:
        MRR value between 0.0 and 1.0. Returns 0.0 if no relevant documents
        are found in the retrieved list.

    Example:
        >>> mean_reciprocal_rank(['a', 'b', 'c'], {'c'})
        0.333  # First relevant doc at rank 3, so 1/3
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def _dcg_at_k(relevance_scores: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at K.

    DCG = sum(rel_i / log2(i + 2)) for i in 0..k-1

    The +2 in the denominator ensures:
    - Position 1 (i=0): discount = log2(2) = 1
    - Position 2 (i=1): discount = log2(3) = 1.58
    - etc.

    Args:
        relevance_scores: List of relevance grades in result order.
        k: Number of positions to consider.

    Returns:
        DCG value (non-negative).
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]) if rel > 0)


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Calculate normalized Discounted Cumulative Gain at K (nDCG@K).

    nDCG measures ranking quality with graded relevance. Unlike binary
    precision/recall, it rewards returning highly relevant documents
    before marginally relevant ones.

    The formula normalizes DCG by the ideal DCG (if results were perfectly
    ordered by relevance), producing a value between 0 and 1.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        relevance_grades: Mapping from doc_id to relevance grade (0-3),
            where 0=not relevant, 1=marginally, 2=relevant, 3=highly relevant.
        k: Number of top results to consider.

    Returns:
        nDCG value between 0.0 and 1.0. Returns 0.0 if there are no
        relevant documents or k <= 0.

    Example:
        >>> ndcg_at_k(['a', 'b', 'c'], {'a': 3, 'b': 1, 'c': 2}, k=3)
        0.936  # Good ranking, highly relevant doc first
    """
    if k <= 0 or not relevance_grades:
        return 0.0

    # Get relevance scores in retrieved order
    retrieved_relevance = [relevance_grades.get(doc_id, 0) for doc_id in retrieved_doc_ids[:k]]
    dcg = _dcg_at_k(retrieved_relevance, k)

    # Ideal DCG: sort all relevance grades descending
    ideal_relevance = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = _dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> float:
    """Calculate Average Precision (AP) for a single query.

    AP is the average of precision values at each position where a relevant
    document is found. It rewards systems that return relevant documents
    early and penalizes those that scatter them throughout the ranking.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        relevant_doc_ids: Set of document IDs that are relevant (ground truth).

    Returns:
        AP value between 0.0 and 1.0. Returns 0.0 if there are no relevant
        documents in the ground truth.

    Example:
        >>> average_precision(['a', 'b', 'c', 'd'], {'a', 'c'})
        0.75  # (1/1 + 2/3) / 2 = (1.0 + 0.667) / 2
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return 0.0

    precision_sum = 0.0
    relevant_found = 0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            relevant_found += 1
            precision_at_rank = relevant_found / rank
            precision_sum += precision_at_rank

    if relevant_found == 0:
        return 0.0

    # Divide by total relevant docs (not just those found)
    return precision_sum / len(relevant_doc_ids)


def compute_all_metrics(
    retrieved_doc_ids: list[str],
    relevance_grades: dict[str, int],
    k_values: list[int] | None = None,
) -> list[MetricResult]:
    """Compute all standard metrics at multiple k values.

    This is a convenience function that computes precision, recall, nDCG,
    MRR, and average precision for a single query's results.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document IDs.
        relevance_grades: Mapping from doc_id to relevance grade (0-3).
        k_values: List of k values to compute metrics at. Defaults to [5, 10, 20].

    Returns:
        List of MetricResult objects for all computed metrics.

    Example:
        >>> results = compute_all_metrics(['a', 'b'], {'a': 3, 'c': 2}, k_values=[5])
        >>> for r in results:
        ...     print(f"{r.name}@{r.k_value}: {r.value:.3f}")
        precision@5: 0.200
        recall@5: 0.500
        ndcg@5: 0.704
        mrr@None: 1.000
        ap@None: 0.500
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # Convert graded relevance to binary set for precision/recall/MRR/AP
    # Documents with grade > 0 are considered relevant
    relevant_doc_ids = {doc_id for doc_id, grade in relevance_grades.items() if grade > 0}

    results: list[MetricResult] = []

    # Compute metrics at each k value
    for k in k_values:
        results.append(
            MetricResult(
                name="precision",
                k_value=k,
                value=precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            )
        )
        results.append(
            MetricResult(
                name="recall",
                k_value=k,
                value=recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            )
        )
        results.append(
            MetricResult(
                name="ndcg",
                k_value=k,
                value=ndcg_at_k(retrieved_doc_ids, relevance_grades, k),
            )
        )

    # MRR and AP don't use k values
    results.append(
        MetricResult(
            name="mrr",
            k_value=None,
            value=mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
        )
    )
    results.append(
        MetricResult(
            name="ap",
            k_value=None,
            value=average_precision(retrieved_doc_ids, relevant_doc_ids),
        )
    )

    return results


__all__ = [
    "collapse_chunks_to_documents",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "average_precision",
    "compute_all_metrics",
]
