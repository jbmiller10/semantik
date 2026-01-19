"""
Tests for SearchService.benchmark_search retry/error semantics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from webui.services.search_service import SearchService


@pytest.fixture()
def search_service(mock_db_session: AsyncMock, mock_collection_repo: AsyncMock) -> SearchService:
    return SearchService(db_session=mock_db_session, collection_repo=mock_collection_repo)


class TestSearchServiceBenchmarkSearch:
    @pytest.mark.asyncio()
    async def test_retries_transient_http_status_then_succeeds(
        self,
        search_service: SearchService,
        mock_collection: MagicMock,
    ) -> None:
        request = httpx.Request("POST", "http://vecpipe.test/search")
        response_500 = httpx.Response(500, request=request, content=b"oops")
        err = httpx.HTTPStatusError("server error", request=request, response=response_500)

        resp1 = MagicMock()
        resp1.raise_for_status.side_effect = err

        resp2 = MagicMock()
        resp2.raise_for_status = MagicMock()
        resp2.json.return_value = {"results": [], "metadata": {"rerank_time_ms": 0}}

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("webui.services.search_service.asyncio.sleep", new=AsyncMock()) as mock_sleep,
            patch("webui.services.search_service.random.uniform", return_value=0.0),
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [resp1, resp2]

            result = await search_service.benchmark_search(
                collection=mock_collection,
                query="test",
                search_mode="dense",
                use_reranker=False,
                top_k=5,
            )

            assert result.total_results == 0
            assert mock_client.post.call_count == 2
            mock_sleep.assert_awaited()

    @pytest.mark.asyncio()
    async def test_does_not_retry_non_transient_http_status(
        self,
        search_service: SearchService,
        mock_collection: MagicMock,
    ) -> None:
        request = httpx.Request("POST", "http://vecpipe.test/search")
        response_400 = httpx.Response(400, request=request, content=b"bad request")
        err = httpx.HTTPStatusError("bad request", request=request, response=response_400)

        resp = MagicMock()
        resp.raise_for_status.side_effect = err

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("webui.services.search_service.asyncio.sleep", new=AsyncMock()) as mock_sleep,
            patch("webui.services.search_service.random.uniform", return_value=0.0),
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = resp

            with pytest.raises(RuntimeError, match="status=400"):
                await search_service.benchmark_search(
                    collection=mock_collection,
                    query="test",
                    search_mode="dense",
                    use_reranker=False,
                    top_k=5,
                )

            assert mock_client.post.call_count == 1
            mock_sleep.assert_not_awaited()

