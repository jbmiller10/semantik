"""Unit tests for CollectionSyncRunRepository."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError
from shared.database.repositories.collection_sync_run_repository import CollectionSyncRunRepository


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
    return CollectionSyncRunRepository(mock_session)


# -----------------------------------------------------------------------------
# Test create
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_create_success(repo, mock_session):
    """Test successful sync run creation."""
    mock_session.flush = AsyncMock()

    result = await repo.create(
        collection_id="col-123",
        triggered_by="scheduler",
        expected_sources=3,
        meta={"reason": "scheduled sync"},
    )

    assert result is not None
    assert result.collection_id == "col-123"
    assert result.triggered_by == "scheduler"
    assert result.expected_sources == 3
    assert result.status == "running"
    assert result.completed_sources == 0
    assert result.failed_sources == 0
    assert result.partial_sources == 0
    mock_session.add.assert_called_once()
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_create_manual_trigger(repo):
    """Test sync run creation with manual trigger."""
    result = await repo.create(
        collection_id="col-123",
        triggered_by="manual",
        expected_sources=1,
    )

    assert result.triggered_by == "manual"


@pytest.mark.asyncio()
async def test_create_database_error(repo, mock_session):
    """Test create raises DatabaseOperationError on error."""
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.create(
            collection_id="col-123",
            triggered_by="scheduler",
            expected_sources=1,
        )


# -----------------------------------------------------------------------------
# Test get_by_id
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_by_id_found(repo, mock_session):
    """Test get_by_id returns sync run when found."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    result = await repo.get_by_id(1)

    assert result == mock_run


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
# Test get_active_run
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_active_run_found(repo, mock_session):
    """Test get_active_run returns running sync run."""
    mock_run = MagicMock()
    mock_run.status = "running"
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    result = await repo.get_active_run("col-123")

    assert result == mock_run


@pytest.mark.asyncio()
async def test_get_active_run_not_found(repo, mock_session):
    """Test get_active_run returns None when no active run."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await repo.get_active_run("col-123")

    assert result is None


@pytest.mark.asyncio()
async def test_get_active_run_database_error(repo, mock_session):
    """Test get_active_run raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.get_active_run("col-123")


# -----------------------------------------------------------------------------
# Test update_source_completion
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_update_source_completion_success(repo, mock_session):
    """Test update_source_completion increments completed_sources."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.completed_sources = 0
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    result = await repo.update_source_completion(1, "success")

    assert result == mock_run
    assert mock_run.completed_sources == 1
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_update_source_completion_failed(repo, mock_session):
    """Test update_source_completion increments failed_sources."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.completed_sources = 0
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.update_source_completion(1, "failed")

    assert mock_run.failed_sources == 1


@pytest.mark.asyncio()
async def test_update_source_completion_partial(repo, mock_session):
    """Test update_source_completion increments partial_sources."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.completed_sources = 0
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.update_source_completion(1, "partial")

    assert mock_run.partial_sources == 1


@pytest.mark.asyncio()
async def test_update_source_completion_unknown_status_treated_as_partial(repo, mock_session):
    """Test unknown status is treated as partial."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.completed_sources = 0
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.update_source_completion(1, "unknown_status")

    assert mock_run.partial_sources == 1


@pytest.mark.asyncio()
async def test_update_source_completion_not_found(repo, mock_session):
    """Test update_source_completion raises EntityNotFoundError."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(EntityNotFoundError):
        await repo.update_source_completion(999, "success")


@pytest.mark.asyncio()
async def test_update_source_completion_database_error(repo, mock_session):
    """Test update_source_completion raises DatabaseOperationError on error."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.completed_sources = 0
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.update_source_completion(1, "success")


# -----------------------------------------------------------------------------
# Test complete_run
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_complete_run_all_success(repo, mock_session):
    """Test complete_run with all sources successful."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.expected_sources = 3
    mock_run.completed_sources = 3
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.complete_run(1)

    assert mock_run.status == "success"
    assert mock_run.completed_at is not None
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_complete_run_all_failed(repo, mock_session):
    """Test complete_run with all sources failed."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.expected_sources = 3
    mock_run.completed_sources = 0
    mock_run.failed_sources = 3
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.complete_run(1, error_summary="All sources failed")

    assert mock_run.status == "failed"
    assert mock_run.error_summary == "All sources failed"


@pytest.mark.asyncio()
async def test_complete_run_partial_with_failures(repo, mock_session):
    """Test complete_run with some failures results in partial status."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.expected_sources = 3
    mock_run.completed_sources = 2
    mock_run.failed_sources = 1
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.complete_run(1)

    assert mock_run.status == "partial"


@pytest.mark.asyncio()
async def test_complete_run_partial_with_partial_sources(repo, mock_session):
    """Test complete_run with partial sources results in partial status."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.expected_sources = 3
    mock_run.completed_sources = 2
    mock_run.failed_sources = 0
    mock_run.partial_sources = 1
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    await repo.complete_run(1)

    assert mock_run.status == "partial"


@pytest.mark.asyncio()
async def test_complete_run_not_found(repo, mock_session):
    """Test complete_run raises EntityNotFoundError."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(EntityNotFoundError):
        await repo.complete_run(999)


@pytest.mark.asyncio()
async def test_complete_run_database_error(repo, mock_session):
    """Test complete_run raises DatabaseOperationError on error."""
    mock_run = MagicMock()
    mock_run.id = 1
    mock_run.expected_sources = 1
    mock_run.completed_sources = 1
    mock_run.failed_sources = 0
    mock_run.partial_sources = 0
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result
    mock_session.flush.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.complete_run(1)


# -----------------------------------------------------------------------------
# Test list_for_collection
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_list_for_collection_returns_runs(repo, mock_session):
    """Test list_for_collection returns runs and count."""
    mock_runs = [MagicMock(), MagicMock()]

    # Mock count
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 2

    # Mock runs
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_runs
    mock_result.scalars.return_value = mock_scalars

    mock_session.execute.side_effect = [mock_count_result, mock_result]

    runs, total = await repo.list_for_collection("col-123", offset=0, limit=50)

    assert runs == mock_runs
    assert total == 2


@pytest.mark.asyncio()
async def test_list_for_collection_empty(repo, mock_session):
    """Test list_for_collection returns empty list when no runs."""
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 0

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result.scalars.return_value = mock_scalars

    mock_session.execute.side_effect = [mock_count_result, mock_result]

    runs, total = await repo.list_for_collection("col-123")

    assert runs == []
    assert total == 0


@pytest.mark.asyncio()
async def test_list_for_collection_database_error(repo, mock_session):
    """Test list_for_collection raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.list_for_collection("col-123")


# -----------------------------------------------------------------------------
# Test get_latest_for_collection
# -----------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_latest_for_collection_found(repo, mock_session):
    """Test get_latest_for_collection returns most recent run."""
    mock_run = MagicMock()
    mock_run.id = 5
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_run
    mock_session.execute.return_value = mock_result

    result = await repo.get_latest_for_collection("col-123")

    assert result == mock_run


@pytest.mark.asyncio()
async def test_get_latest_for_collection_not_found(repo, mock_session):
    """Test get_latest_for_collection returns None when no runs."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    result = await repo.get_latest_for_collection("col-123")

    assert result is None


@pytest.mark.asyncio()
async def test_get_latest_for_collection_database_error(repo, mock_session):
    """Test get_latest_for_collection raises DatabaseOperationError on error."""
    mock_session.execute.side_effect = Exception("DB error")

    with pytest.raises(DatabaseOperationError):
        await repo.get_latest_for_collection("col-123")
