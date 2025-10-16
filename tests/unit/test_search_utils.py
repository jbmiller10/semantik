"""Unit tests for search utility helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from packages.vecpipe.search_utils import search_qdrant


@pytest.mark.asyncio()
async def test_search_qdrant_closes_client() -> None:
    """search_qdrant should close the AsyncQdrantClient after use."""

    with patch("packages.vecpipe.search_utils.AsyncQdrantClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.search = AsyncMock(
            return_value=[
                SimpleNamespace(id="doc-1", score=0.9, payload={"path": "/tmp/doc-1", "chunk_id": "c1"})
            ]
        )
        mock_client.aclose = AsyncMock()

        results = await search_qdrant(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="test",
            query_vector=[0.1, 0.2, 0.3],
            k=5,
            with_payload=True,
        )

        mock_client_cls.assert_called_once_with(url="http://localhost:6333")
        mock_client.search.assert_awaited_once()
        mock_client.aclose.assert_awaited_once()

        assert results == [
            {"id": "doc-1", "score": 0.9, "payload": {"path": "/tmp/doc-1", "chunk_id": "c1"}}
        ]
