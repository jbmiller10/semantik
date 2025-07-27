"""
Test fixtures for search reranking functionality.

Provides reusable mocks and test data for reranking tests.
"""

from typing import Any
from unittest.mock import MagicMock

from packages.shared.database.models import Collection, CollectionStatus


def create_mock_collection(
    collection_id: str = "123e4567-e89b-12d3-a456-426614174000",
    name: str = "Test Collection",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    quantization: str = "float16",
    status: CollectionStatus = CollectionStatus.READY,
) -> MagicMock:
    """Create a mock collection with default values."""
    collection = MagicMock(spec=Collection)
    collection.id = collection_id
    collection.name = name
    collection.vector_store_name = f"qdrant_{name.lower().replace(' ', '_')}"
    collection.embedding_model = embedding_model
    collection.quantization = quantization
    collection.status = status
    collection.document_count = 100
    return collection


def create_search_result(
    doc_id: str,
    chunk_id: str,
    score: float,
    content: str,
    path: str,
    reranked_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a search result dictionary."""
    result = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "score": score,
        "content": content,
        "path": path,
        "metadata": metadata or {},
    }
    if reranked_score is not None:
        result["reranked_score"] = reranked_score
    return result


def create_vecpipe_response(
    results: list[dict[str, Any]],
    processing_time_ms: float = 100,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a mock vecpipe service response."""
    if error:
        return {"error": error, "detail": error}

    return {
        "results": results,
        "processing_time_ms": processing_time_ms,
    }


def create_multi_collection_search_response(
    results: list[dict[str, Any]],
    collection_details: list[dict[str, Any]],
    processing_time: float = 0.1,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a multi-collection search response."""
    return {
        "results": results,
        "metadata": {
            "total_results": len(results),
            "collections_searched": len(collection_details),
            "collection_details": collection_details,
            "processing_time": processing_time,
            "errors": errors,
        },
    }


# Sample reranking test scenarios
RERANKING_TEST_SCENARIOS = {
    "basic_reranking": {
        "description": "Basic reranking with score improvement",
        "original_scores": [0.7, 0.65, 0.6],
        "reranked_scores": [0.95, 0.88, 0.82],
        "expected_order": [0, 1, 2],  # Same order but higher scores
    },
    "reordering": {
        "description": "Reranking changes result order",
        "original_scores": [0.8, 0.75, 0.85],
        "reranked_scores": [0.75, 0.95, 0.88],
        "expected_order": [1, 2, 0],  # Order changed after reranking
    },
    "mixed_collections": {
        "description": "Reranking across multiple collections",
        "collections": ["collection1", "collection2"],
        "original_scores": [0.9, 0.85, 0.88, 0.82],
        "reranked_scores": [0.85, 0.98, 0.82, 0.95],
        "expected_order": [1, 3, 0, 2],
    },
}


# Reranker model configurations for testing
RERANKER_MODELS = {
    "Qwen/Qwen3-Reranker-0.6B": {
        "size": "0.6B",
        "performance": "fastest",
        "memory_usage": "low",
        "typical_latency_ms": 50,
    },
    "Qwen/Qwen3-Reranker-4B": {
        "size": "4B",
        "performance": "balanced",
        "memory_usage": "medium",
        "typical_latency_ms": 100,
    },
    "Qwen/Qwen3-Reranker-8B": {
        "size": "8B",
        "performance": "most_accurate",
        "memory_usage": "high",
        "typical_latency_ms": 200,
    },
}


# Error scenarios for testing
RERANKING_ERROR_SCENARIOS = {
    "insufficient_memory": {
        "status_code": 507,
        "error": "insufficient_memory",
        "message": "Insufficient GPU memory for reranking",
        "suggestion": "Try using a smaller model or different quantization",
    },
    "model_not_found": {
        "status_code": 404,
        "error": "model_not_found",
        "message": "Reranker model not found",
        "suggestion": "Check model name or use default model",
    },
    "timeout": {
        "status_code": 504,
        "error": "timeout",
        "message": "Reranking operation timed out",
        "suggestion": "Try with fewer results or simpler model",
    },
}
