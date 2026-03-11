"""Tests for source stats gathering."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.models import CollectionSource


class TestGetSourceStats:
    """Test get_source_stats function."""

    @pytest.fixture()
    def mock_source(self) -> CollectionSource:
        """Create a mock collection source."""
        source = MagicMock(spec=CollectionSource)
        source.id = 42
        source.source_path = "/data/docs"
        source.source_type = "directory"
        source.source_config = {"path": "/data/docs"}
        return source

    @pytest.mark.asyncio()
    async def test_get_source_stats_basic(self, mock_source: CollectionSource) -> None:
        """Returns basic source info."""
        from webui.services.assisted_flow.source_stats import get_source_stats

        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = (mock_source, 1)
        session.execute = AsyncMock(return_value=result)

        stats = await get_source_stats(session, user_id=1, source_id=mock_source.id)

        assert stats["source_name"] == "/data/docs"
        assert stats["source_type"] == "directory"
        assert stats["source_path"] == "/data/docs"

    @pytest.mark.asyncio()
    async def test_get_source_stats_not_found(self) -> None:
        """Raises error when source not found."""
        from shared.database.exceptions import EntityNotFoundError
        from webui.services.assisted_flow.source_stats import get_source_stats

        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = None
        session.execute = AsyncMock(return_value=result)

        with pytest.raises(EntityNotFoundError):
            await get_source_stats(session, user_id=1, source_id=999)

    @pytest.mark.asyncio()
    async def test_get_source_stats_access_denied(self, mock_source: CollectionSource) -> None:
        """Raises error when user does not own the source."""
        from shared.database.exceptions import AccessDeniedError
        from webui.services.assisted_flow.source_stats import get_source_stats

        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = (mock_source, 999)
        session.execute = AsyncMock(return_value=result)

        with pytest.raises(AccessDeniedError):
            await get_source_stats(session, user_id=1, source_id=mock_source.id)
