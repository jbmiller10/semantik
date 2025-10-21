"""Unit tests for the collection metadata helpers."""

from unittest.mock import MagicMock, patch

from shared.database.collection_metadata import store_collection_metadata


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
