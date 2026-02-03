"""Tests for source stats gathering."""

from unittest.mock import AsyncMock, MagicMock, patch

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

        with patch("webui.services.assisted_flow.source_stats.CollectionSourceRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            # Make get_by_id return an awaitable
            mock_repo.get_by_id = AsyncMock(return_value=mock_source)

            stats = await get_source_stats(session, mock_source.id)

        assert stats["source_name"] == "/data/docs"
        assert stats["source_type"] == "directory"
        assert stats["source_path"] == "/data/docs"

    @pytest.mark.asyncio()
    async def test_get_source_stats_not_found(self) -> None:
        """Raises error when source not found."""
        from shared.database.exceptions import EntityNotFoundError
        from webui.services.assisted_flow.source_stats import get_source_stats

        session = AsyncMock()

        with patch("webui.services.assisted_flow.source_stats.CollectionSourceRepository") as mock_repo_cls:
            mock_repo = mock_repo_cls.return_value
            # Make get_by_id return an awaitable that returns None
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(EntityNotFoundError):
                await get_source_stats(session, 999)
