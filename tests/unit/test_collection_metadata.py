"""Tests for collection metadata helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from packages.shared.database.collection_metadata import (
    METADATA_COLLECTION,
    get_collection_metadata,
    restamp_collection_metadata,
    store_collection_metadata,
)


@pytest.fixture()
def mock_qdrant_client():
    """Create a mutable mock for the Qdrant client."""

    mock_client = MagicMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name=METADATA_COLLECTION)])
    return mock_client


def test_store_collection_metadata_uses_collection_name_as_id(mock_qdrant_client):
    """store_collection_metadata should upsert using the collection name as the point ID."""

    store_collection_metadata(
        mock_qdrant_client,
        collection_name="research_docs",
        model_name="BAAI/bge-base-en",
        quantization="float16",
        vector_dim=1024,
        chunk_size=512,
        chunk_overlap=128,
    )

    mock_qdrant_client.upsert.assert_called_once()
    call_kwargs = mock_qdrant_client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == METADATA_COLLECTION
    point = call_kwargs["points"][0]
    assert point.id == "research_docs"
    assert point.payload["collection_name"] == "research_docs"


def test_get_collection_metadata_retrieves_by_id(mock_qdrant_client):
    """Metadata retrieval should look up by deterministic point ID."""

    payload = {
        "collection_name": "project_alpha",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "quantization": "float16",
    }
    mock_qdrant_client.retrieve.return_value = [SimpleNamespace(payload=payload)]

    result = get_collection_metadata(mock_qdrant_client, "project_alpha")

    mock_qdrant_client.retrieve.assert_called_once_with(
        collection_name=METADATA_COLLECTION, ids=["project_alpha"]
    )
    assert result == payload


def test_get_collection_metadata_falls_back_to_legacy_entries(mock_qdrant_client):
    """If metadata is stored under a legacy random point ID, the fallback scroll should find it."""

    mock_qdrant_client.retrieve.return_value = []
    legacy_payload = {
        "collection_name": "legacy_docs",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }
    mock_qdrant_client.scroll.side_effect = [
        ([SimpleNamespace(id="random-id", payload=legacy_payload)], None),
    ]

    result = get_collection_metadata(mock_qdrant_client, "legacy_docs")

    assert result == legacy_payload
    mock_qdrant_client.scroll.assert_called_once()


def test_restamp_collection_metadata_upserts_deterministic_ids(mock_qdrant_client):
    """restamp_collection_metadata should upsert new records for legacy entries."""

    mock_qdrant_client.scroll.side_effect = [
        ([SimpleNamespace(id="legacy-id", payload={"collection_name": "col-one"})], None),
    ]

    migrated = restamp_collection_metadata(mock_qdrant_client, batch_size=10)

    assert migrated == 1
    mock_qdrant_client.upsert.assert_called()
    upsert_points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    assert upsert_points[0].id == "col-one"
