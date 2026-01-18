"""Unit tests for CollectionSourceRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.repositories.collection_source_repository import CollectionSourceRepository


@pytest.fixture()
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.scalar = AsyncMock()
    return session


@pytest.fixture()
def repo(mock_session):
    """Create repository with mock session."""
    return CollectionSourceRepository(mock_session)


# -----------------------------------------------------------------------------
# Test create
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_create_success(repo, mock_session):
    """Test successful source creation."""
    mock_session.flush = AsyncMock()

    result = await repo.create(
        collection_id="col-123",
        source_type="directory",
        source_path="/data/docs",
        source_config={"recursive": True},
        meta={"label": "documents"},
    )

    assert result is not None
    assert result.collection_id == "col-123"
    assert result.source_type == "directory"
    assert result.source_path == "/data/docs"
    mock_session.add.assert_called_once()
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_create_missing_collection_id_raises_validation_error(repo):
    """Test that missing collection_id raises ValidationError."""
    with pytest.raises(ValidationError):
        await repo.create(
            collection_id="",
            source_type="directory",
            source_path="/data/docs",
        )


@pytest.mark.asyncio()
async def test_create_missing_source_type_raises_validation_error(repo):
    """Test that missing source_type raises ValidationError."""
    with pytest.raises(ValidationError):
        await repo.create(
            collection_id="col-123",
            source_type="",
            source_path="/data/docs",
        )


@pytest.mark.asyncio()
async def test_create_missing_source_path_raises_validation_error(repo):
    """Test that missing source_path raises ValidationError."""
    with pytest.raises(ValidationError):
        await repo.create(
            collection_id="col-123",
            source_type="directory",
            source_path="",
        )


@pytest.mark.asyncio()
async def test_create_integrity_error_raises_already_exists(repo, mock_session):
    """Test that IntegrityError raises EntityAlreadyExistsError."""
    mock_session.flush.side_effect = IntegrityError("duplicate", None, None)

    with pytest.raises(EntityAlreadyExistsError):
        await repo.create(
            collection_id="col-123",
            source_type="directory",
            source_path="/data/docs",
        )


@pytest.mark.asyncio()
async def test_create_database_error_raises_operation_error(repo, mock_session):
    """Test that database errors raise DatabaseOperationError."""
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.create(
            collection_id="col-123",
            source_type="directory",
            source_path="/data/docs",
        )


# -----------------------------------------------------------------------------
# Test get_by_id
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_by_id_found(repo, mock_session):
    """Test get_by_id returns source when found."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result = await repo.get_by_id(1)

    assert result == mock_source
    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_by_id_not_found(repo, mock_session):
    """Test get_by_id returns None when not found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await repo.get_by_id(999)

    assert result is None


@pytest.mark.asyncio()
async def test_get_by_id_database_error(repo, mock_session):
    """Test get_by_id raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.get_by_id(1)


# -----------------------------------------------------------------------------
# Test get_by_collection_and_path
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_by_collection_and_path_found(repo, mock_session):
    """Test get_by_collection_and_path returns source when found."""
    mock_source = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result = await repo.get_by_collection_and_path("col-123", "/data/docs")

    assert result == mock_source


@pytest.mark.asyncio()
async def test_get_by_collection_and_path_not_found(repo, mock_session):
    """Test get_by_collection_and_path returns None when not found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await repo.get_by_collection_and_path("col-123", "/nonexistent")

    assert result is None


@pytest.mark.asyncio()
async def test_get_by_collection_and_path_database_error(repo, mock_session):
    """Test get_by_collection_and_path raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.get_by_collection_and_path("col-123", "/data/docs")


# -----------------------------------------------------------------------------
# Test get_or_create
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_or_create_existing_no_update(repo, mock_session):
    """Test get_or_create returns existing source without update."""
    mock_source = MagicMock()
    mock_source.source_type = "directory"
    mock_source.source_config = {"recursive": True}
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result, is_new = await repo.get_or_create(
        collection_id="col-123",
        source_type="directory",
        source_path="/data/docs",
        source_config={"recursive": True},
    )

    assert result == mock_source
    assert is_new is False
    # No flush should be called since nothing changed
    assert mock_session.flush.await_count == 0


@pytest.mark.asyncio()
async def test_get_or_create_existing_updates_type(repo, mock_session):
    """Test get_or_create updates source_type when different."""
    mock_source = MagicMock()
    mock_source.source_type = "directory"
    mock_source.source_config = {}
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result, is_new = await repo.get_or_create(
        collection_id="col-123",
        source_type="web",
        source_path="/data/docs",
    )

    assert result == mock_source
    assert is_new is False
    assert mock_source.source_type == "web"
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_or_create_existing_updates_config(repo, mock_session):
    """Test get_or_create updates source_config when different."""
    mock_source = MagicMock()
    mock_source.source_type = "directory"
    mock_source.source_config = {"old": "config"}
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result, is_new = await repo.get_or_create(
        collection_id="col-123",
        source_type="directory",
        source_path="/data/docs",
        source_config={"new": "config"},
    )

    assert result == mock_source
    assert is_new is False
    assert mock_source.source_config == {"new": "config"}
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_or_create_creates_new(repo, mock_session):
    """Test get_or_create creates new source when not found."""
    # First call returns None (not found)
    mock_result_none = MagicMock()
    mock_result_none.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result_none

    result, is_new = await repo.get_or_create(
        collection_id="col-123",
        source_type="directory",
        source_path="/data/docs",
    )

    assert result is not None
    assert is_new is True
    mock_session.add.assert_called_once()


@pytest.mark.asyncio()
async def test_get_or_create_database_error(repo, mock_session):
    """Test get_or_create raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.get_or_create(
            collection_id="col-123",
            source_type="directory",
            source_path="/data/docs",
        )


# -----------------------------------------------------------------------------
# Test list_by_collection
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_list_by_collection_returns_sources(repo, mock_session):
    """Test list_by_collection returns sources and count."""
    mock_sources = [MagicMock(), MagicMock()]

    # Mock total count
    mock_session.scalar = AsyncMock(return_value=2)

    # Mock sources
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_sources
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    sources, total = await repo.list_by_collection("col-123", offset=0, limit=50)

    assert sources == mock_sources
    assert total == 2


@pytest.mark.asyncio()
async def test_list_by_collection_empty(repo, mock_session):
    """Test list_by_collection returns empty list when no sources."""
    mock_session.scalar = AsyncMock(return_value=0)
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    sources, total = await repo.list_by_collection("col-123")

    assert sources == []
    assert total == 0


@pytest.mark.asyncio()
async def test_list_by_collection_database_error(repo, mock_session):
    """Test list_by_collection raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.list_by_collection("col-123")


# -----------------------------------------------------------------------------
# Test update_stats
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_update_stats_success(repo, mock_session):
    """Test update_stats successfully updates stats."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result = await repo.update_stats(
        source_id=1,
        document_count=100,
        size_bytes=1024000,
        last_indexed_at=datetime.now(UTC),
    )

    assert result == mock_source
    assert mock_source.document_count == 100
    assert mock_source.size_bytes == 1024000
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_update_stats_negative_document_count_raises_validation_error(repo):
    """Test that negative document_count raises ValidationError."""
    with pytest.raises(ValidationError):
        await repo.update_stats(source_id=1, document_count=-1)


@pytest.mark.asyncio()
async def test_update_stats_negative_size_bytes_raises_validation_error(repo):
    """Test that negative size_bytes raises ValidationError."""
    with pytest.raises(ValidationError):
        await repo.update_stats(source_id=1, size_bytes=-1)


@pytest.mark.asyncio()
async def test_update_stats_not_found_raises_entity_not_found(repo, mock_session):
    """Test update_stats raises EntityNotFoundError when source not found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(EntityNotFoundError):
        await repo.update_stats(source_id=999, document_count=10)


@pytest.mark.asyncio()
async def test_update_stats_database_error(repo, mock_session):
    """Test update_stats raises DatabaseOperationError on error."""
    mock_source = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.update_stats(source_id=1, document_count=10)


# -----------------------------------------------------------------------------
# Test delete
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_delete_success(repo, mock_session):
    """Test successful source deletion."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    await repo.delete(1)

    # Should have two execute calls: get_by_id + delete
    assert mock_session.execute.await_count >= 1
    mock_session.flush.assert_awaited()


@pytest.mark.asyncio()
async def test_delete_not_found_raises_entity_not_found(repo, mock_session):
    """Test delete raises EntityNotFoundError when source not found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(EntityNotFoundError):
        await repo.delete(999)


@pytest.mark.asyncio()
async def test_delete_database_error(repo, mock_session):
    """Test delete raises DatabaseOperationError on error."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    # First execute (get_by_id) succeeds, second (delete) fails
    mock_session.execute.side_effect = [mock_result, Exception("DB error")]

    with pytest.raises(DatabaseOperationError):
        await repo.delete(1)


# -----------------------------------------------------------------------------
# Test update
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_update_success(repo, mock_session):
    """Test successful source update."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_source.source_config = {}
    mock_source.meta = {}
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result = await repo.update(
        source_id=1,
        source_config={"updated": "config"},
        meta={"new": "meta"},
    )

    assert result == mock_source
    assert mock_source.source_config == {"updated": "config"}
    assert mock_source.meta == {"new": "meta"}
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_update_not_found_raises_entity_not_found(repo, mock_session):
    """Test update raises EntityNotFoundError when source not found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(EntityNotFoundError):
        await repo.update(source_id=999, source_config={})


@pytest.mark.asyncio()
async def test_update_database_error(repo, mock_session):
    """Test update raises DatabaseOperationError on error."""
    mock_source = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.update(source_id=1, source_config={})


# -----------------------------------------------------------------------------
# Test update_sync_status
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_update_sync_status_success(repo, mock_session):
    """Test successful sync status update."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    started = datetime.now(UTC)
    completed = datetime.now(UTC)

    result = await repo.update_sync_status(
        source_id=1,
        status="success",
        started_at=started,
        completed_at=completed,
    )

    assert result == mock_source
    assert mock_source.last_run_status == "success"
    assert mock_source.last_error is None
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_update_sync_status_failed_with_error(repo, mock_session):
    """Test sync status update with failure and error message."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result = await repo.update_sync_status(
        source_id=1,
        status="failed",
        error="Connection timeout",
    )

    assert result == mock_source
    assert mock_source.last_run_status == "failed"
    assert mock_source.last_error == "Connection timeout"


@pytest.mark.asyncio()
async def test_update_sync_status_partial_with_error(repo, mock_session):
    """Test sync status update with partial status and error."""
    mock_source = MagicMock()
    mock_source.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result

    result = await repo.update_sync_status(
        source_id=1,
        status="partial",
        error="Some files skipped",
    )

    assert result == mock_source
    assert mock_source.last_run_status == "partial"
    assert mock_source.last_error == "Some files skipped"


@pytest.mark.asyncio()
async def test_update_sync_status_invalid_status_raises_validation_error(repo):
    """Test that invalid status raises ValidationError."""
    with pytest.raises(ValidationError):
        await repo.update_sync_status(source_id=1, status="invalid")


@pytest.mark.asyncio()
async def test_update_sync_status_not_found_raises_entity_not_found(repo, mock_session):
    """Test update_sync_status raises EntityNotFoundError when not found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(EntityNotFoundError):
        await repo.update_sync_status(source_id=999, status="success")


@pytest.mark.asyncio()
async def test_update_sync_status_database_error(repo, mock_session):
    """Test update_sync_status raises DatabaseOperationError on error."""
    mock_source = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    mock_session.execute.return_value = mock_result
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.update_sync_status(source_id=1, status="success")
