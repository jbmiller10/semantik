"""Unit tests for the collection metadata helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.collection_metadata import (
    get_collection_metadata,
    get_collection_metadata_async,
    store_collection_metadata,
)


def _build_qdrant_mock() -> MagicMock:
    client = MagicMock()
    collections_obj = MagicMock()
    collections_obj.collections = []
    client.get_collections.return_value = collections_obj
    client.upsert = MagicMock()
    return client


def test_store_collection_metadata_calls_ensure_by_default() -> None:
    client = _build_qdrant_mock()

    with patch("shared.database.collection_metadata.ensure_metadata_collection") as mock_ensure:
        store_collection_metadata(
            qdrant=client,
            collection_name="col_test",
            model_name="model",
            quantization="float16",
            vector_dim=1536,
        )

    mock_ensure.assert_called_once_with(client)
    client.upsert.assert_called_once()


def test_store_collection_metadata_skip_ensure_flag() -> None:
    client = _build_qdrant_mock()

    with patch("shared.database.collection_metadata.ensure_metadata_collection") as mock_ensure:
        store_collection_metadata(
            qdrant=client,
            collection_name="col_test",
            model_name="model",
            quantization="float16",
            vector_dim=1536,
            ensure=False,
        )

    mock_ensure.assert_not_called()
    client.upsert.assert_called_once()


def test_get_collection_metadata_returns_payload_from_scroll() -> None:
    """Test that get_collection_metadata returns the payload from scroll results."""
    client = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {
        "collection_name": "col_test",
        "model_name": "test-model",
        "quantization": "float16",
        "vector_dim": 1536,
        "instruction": "test instruction",
    }
    client.scroll.return_value = ([mock_point], None)

    result = get_collection_metadata(client, "col_test")

    assert result is not None
    assert result["collection_name"] == "col_test"
    assert result["model_name"] == "test-model"
    assert result["quantization"] == "float16"
    assert result["vector_dim"] == 1536
    assert result["instruction"] == "test instruction"

    # Verify scroll was called with correct filter
    client.scroll.assert_called_once()
    call_kwargs = client.scroll.call_args.kwargs
    assert call_kwargs["collection_name"] == "_collection_metadata"
    assert call_kwargs["limit"] == 1
    assert call_kwargs["with_payload"] is True
    # Verify filter has the correct structure
    scroll_filter = call_kwargs["scroll_filter"]
    assert len(scroll_filter.must) == 1
    assert scroll_filter.must[0].key == "collection_name"


def test_get_collection_metadata_returns_none_when_no_results() -> None:
    """Test that get_collection_metadata returns None when scroll finds no matches."""
    client = MagicMock()
    client.scroll.return_value = ([], None)

    result = get_collection_metadata(client, "nonexistent_collection")

    assert result is None


def test_get_collection_metadata_returns_none_on_exception() -> None:
    """Test that get_collection_metadata returns None when an exception occurs."""
    client = MagicMock()
    client.scroll.side_effect = Exception("Qdrant connection error")

    result = get_collection_metadata(client, "col_test")

    assert result is None


@pytest.mark.asyncio()
async def test_get_collection_metadata_async_returns_payload() -> None:
    """Test that get_collection_metadata_async returns the payload from scroll results."""
    client = AsyncMock()
    mock_point = MagicMock()
    mock_point.payload = {
        "collection_name": "col_test",
        "model_name": "async-model",
        "quantization": "int8",
        "vector_dim": 768,
    }
    client.scroll.return_value = ([mock_point], None)

    result = await get_collection_metadata_async(client, "col_test")

    assert result is not None
    assert result["model_name"] == "async-model"
    assert result["quantization"] == "int8"
    client.scroll.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_collection_metadata_async_returns_none_when_no_results() -> None:
    """Test that get_collection_metadata_async returns None when scroll finds no matches."""
    client = AsyncMock()
    client.scroll.return_value = ([], None)

    result = await get_collection_metadata_async(client, "nonexistent")

    assert result is None


@pytest.mark.asyncio()
async def test_get_collection_metadata_async_returns_none_on_exception() -> None:
    """Test that get_collection_metadata_async returns None when an exception occurs."""
    client = AsyncMock()
    client.scroll.side_effect = Exception("Async Qdrant error")

    result = await get_collection_metadata_async(client, "col_test")

    assert result is None
