"""Tests for LLM usage repository."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.exceptions import DatabaseOperationError
from shared.database.repositories.llm_usage_repository import LLMUsageRepository, UsageSummary


class TestUsageSummary:
    """Tests for UsageSummary dataclass."""

    def test_usage_summary_creation(self):
        """UsageSummary can be created with all fields."""
        summary = UsageSummary(
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            by_feature={"hyde": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75, "count": 1}},
            by_provider={"anthropic": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "count": 2}},
            event_count=2,
            period_days=30,
        )

        assert summary.total_input_tokens == 100
        assert summary.total_output_tokens == 50
        assert summary.total_tokens == 150
        assert summary.event_count == 2
        assert summary.period_days == 30
        assert "hyde" in summary.by_feature
        assert "anthropic" in summary.by_provider

    def test_usage_summary_is_frozen(self):
        """UsageSummary is immutable."""
        summary = UsageSummary(
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            by_feature={},
            by_provider={},
            event_count=0,
            period_days=30,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            summary.total_tokens = 200  # type: ignore


class TestLLMUsageRepository:
    """Tests for LLMUsageRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture()
    def repo(self, mock_session):
        """Create repository with mocked session."""
        return LLMUsageRepository(mock_session)

    # =========================================================================
    # Record Usage Tests
    # =========================================================================

    async def test_record_usage_creates_event(self, repo, mock_session):
        """record_usage creates an LLMUsageEvent."""
        mock_session.execute.return_value = MagicMock()

        result = await repo.record_usage(
            user_id=123,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=50,
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        assert result is not None

    async def test_record_usage_with_optional_fields(self, repo, mock_session):
        """record_usage accepts optional operation and collection IDs."""
        mock_session.execute.return_value = MagicMock()

        result = await repo.record_usage(
            user_id=123,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="high",
            feature="summary",
            input_tokens=500,
            output_tokens=200,
            operation_id=456,
            collection_id="coll-uuid",
            request_metadata={"key": "value"},
        )

        assert result is not None
        mock_session.add.assert_called_once()

    async def test_record_usage_db_error(self, repo, mock_session):
        """record_usage wraps database errors."""
        mock_session.add.side_effect = Exception("DB error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.record_usage(
                user_id=123,
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                quality_tier="low",
                feature="hyde",
                input_tokens=100,
                output_tokens=50,
            )

        assert "record" in str(exc_info.value)

    # =========================================================================
    # Usage Summary Tests
    # =========================================================================

    async def test_get_user_usage_summary(self, repo, mock_session):
        """get_user_usage_summary returns aggregated data."""
        # Mock totals query result
        mock_totals_row = MagicMock()
        mock_totals_row.input = 1000
        mock_totals_row.output = 500
        mock_totals_row.cnt = 10

        # Mock feature breakdown query result
        mock_feature_row = MagicMock()
        mock_feature_row.feature = "hyde"
        mock_feature_row.input = 400
        mock_feature_row.output = 200
        mock_feature_row.cnt = 5

        # Mock provider breakdown query result
        mock_provider_row = MagicMock()
        mock_provider_row.provider = "anthropic"
        mock_provider_row.input = 1000
        mock_provider_row.output = 500
        mock_provider_row.cnt = 10

        # Setup mock session to return different results for different queries
        mock_results = [
            MagicMock(one=MagicMock(return_value=mock_totals_row)),  # totals
            MagicMock(all=MagicMock(return_value=[mock_feature_row])),  # features
            MagicMock(all=MagicMock(return_value=[mock_provider_row])),  # providers
        ]
        mock_session.execute = AsyncMock(side_effect=mock_results)

        result = await repo.get_user_usage_summary(user_id=123, days=30)

        assert isinstance(result, UsageSummary)
        assert result.total_input_tokens == 1000
        assert result.total_output_tokens == 500
        assert result.total_tokens == 1500
        assert result.event_count == 10
        assert result.period_days == 30

    async def test_get_user_usage_summary_all_time(self, repo, mock_session):
        """get_user_usage_summary with days=0 returns all-time data."""
        mock_totals_row = MagicMock()
        mock_totals_row.input = 5000
        mock_totals_row.output = 2500
        mock_totals_row.cnt = 100

        mock_results = [
            MagicMock(one=MagicMock(return_value=mock_totals_row)),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(all=MagicMock(return_value=[])),
        ]
        mock_session.execute = AsyncMock(side_effect=mock_results)

        result = await repo.get_user_usage_summary(user_id=123, days=0)

        assert result.period_days == 0
        assert result.total_tokens == 7500

    # =========================================================================
    # Get Usage by Feature Tests
    # =========================================================================

    async def test_get_usage_by_feature(self, repo, mock_session):
        """get_usage_by_feature returns filtered events."""
        mock_events = [MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_events
        mock_session.execute.return_value = mock_result

        result = await repo.get_usage_by_feature(
            user_id=123,
            feature="hyde",
            days=30,
            limit=100,
        )

        assert result == mock_events

    async def test_get_usage_by_feature_all_time(self, repo, mock_session):
        """get_usage_by_feature with days=0 returns all events."""
        mock_events = [MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_events
        mock_session.execute.return_value = mock_result

        result = await repo.get_usage_by_feature(
            user_id=123,
            feature="summary",
            days=0,
        )

        assert result == mock_events

    # =========================================================================
    # Get Total Tokens Tests
    # =========================================================================

    async def test_get_total_tokens(self, repo, mock_session):
        """get_total_tokens returns token counts."""
        mock_row = MagicMock()
        mock_row.input = 1000
        mock_row.output = 500

        mock_result = MagicMock()
        mock_result.one.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await repo.get_total_tokens(user_id=123, days=30)

        assert result == {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        }

    async def test_get_total_tokens_empty(self, repo, mock_session):
        """get_total_tokens returns zeros when no events."""
        mock_row = MagicMock()
        mock_row.input = 0
        mock_row.output = 0

        mock_result = MagicMock()
        mock_result.one.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await repo.get_total_tokens(user_id=123)

        assert result["total_tokens"] == 0

    # =========================================================================
    # Get Recent Events Tests
    # =========================================================================

    async def test_get_recent_events(self, repo, mock_session):
        """get_recent_events returns most recent events."""
        mock_events = [MagicMock(), MagicMock(), MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_events
        mock_session.execute.return_value = mock_result

        result = await repo.get_recent_events(user_id=123, limit=10)

        assert result == mock_events

    async def test_get_recent_events_respects_limit(self, repo, mock_session):
        """get_recent_events respects limit parameter."""
        mock_events = [MagicMock()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_events
        mock_session.execute.return_value = mock_result

        result = await repo.get_recent_events(user_id=123, limit=5)

        assert len(result) <= 5
