"""Integration-style test for staging collection lifecycle using the unified manager API."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from packages.shared.managers.qdrant_manager import QdrantManager


def _mock_qdrant_client(staging_name: str) -> MagicMock:
    client = MagicMock()
    client.create_collection.return_value = None
    client.get_collection.return_value = MagicMock(vectors_count=0, payload_schema={})
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name=staging_name)])
    return client


@patch("packages.shared.managers.qdrant_manager.datetime")
def test_staging_create_list_delete(mock_datetime) -> None:
    fixed_time = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    mock_datetime.now.return_value = fixed_time
    mock_datetime.strptime = datetime.strptime  # delegate parsing for _is_staging_collection_old

    base_name = "collection_abc"
    expected_staging = "staging_collection_abc_20240102_030405"

    client = _mock_qdrant_client(expected_staging)
    manager = QdrantManager(client)

    staging_name = manager.create_staging_collection(base_name=base_name, vector_size=128)
    assert staging_name == expected_staging
    client.create_collection.assert_called_once()

    # list + cleanup behavior
    assert manager.list_collections() == [expected_staging]

    with patch.object(manager, "_is_staging_collection_old", return_value=True), patch(
        "packages.shared.managers.qdrant_manager.time.sleep"
    ):
        deleted = manager.cleanup_orphaned_collections(active_collections=[], dry_run=False)

    assert deleted == [expected_staging]
    client.delete_collection.assert_called_once_with(expected_staging)
