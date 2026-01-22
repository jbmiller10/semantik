"""
Shared search utilities for both search API and web UI
"""

import logging

from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)


async def search_qdrant(
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    query_vector: list[float],
    k: int,
    with_payload: bool = True,
) -> list[dict]:
    """
    Perform vector search in Qdrant

    Args:
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        collection_name: Collection to search in
        query_vector: Query embedding vector
        k: Number of results to return
        with_payload: Whether to include payload in results

    Returns:
        List of search results from Qdrant
    """
    # Prefer the globally injected SDK client when available (tests patch this)
    try:
        from vecpipe.search import state as search_state

        sdk_client = getattr(search_state, "sdk_client", None)
        if sdk_client is not None:
            results = await sdk_client.search(
                collection_name=collection_name, query_vector=query_vector, limit=k, with_payload=with_payload
            )
            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload if with_payload else None,
                }
                for point in results
            ]
    except Exception:
        # Fall through to default client
        pass

    # Fallback: create ad-hoc client and ensure proper cleanup
    from vecpipe.search.metrics import qdrant_ad_hoc_client_total

    client = AsyncQdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
    qdrant_ad_hoc_client_total.labels(location="search_utils").inc()
    try:
        results = await client.search(
            collection_name=collection_name, query_vector=query_vector, limit=k, with_payload=with_payload
        )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload if with_payload else None}
            for point in results
        ]
    finally:
        await client.close()


def parse_search_results(qdrant_results: list[dict]) -> list[dict]:
    """
    Parse Qdrant search results into a standard format

    Args:
        qdrant_results: Raw results from Qdrant

    Returns:
        List of parsed results with path, chunk_id, score, etc.
    """
    results = []
    for point in qdrant_results:
        payload = point.get("payload", {})
        result = {
            "path": payload.get("path", ""),
            "chunk_id": payload.get("chunk_id", ""),
            "score": point.get("score", 0.0),
            "doc_id": payload.get("doc_id"),
            "content": payload.get("content"),
            "metadata": payload.get("metadata"),
        }
        results.append(result)
    return results
