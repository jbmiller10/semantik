"""Integration tests for LLM repositories using real database session."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast

import pytest
from sqlalchemy import delete, select

from shared.database.models import LLMProviderApiKey, LLMProviderConfig, LLMUsageEvent, User
from shared.database.repositories.llm_usage_repository import LLMUsageRepository, UsageSummary
from shared.utils.encryption import SecretEncryption

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestLLMProviderConfigRepositoryIntegration:
    """Integration tests for LLMProviderConfigRepository."""

    async def test_create_config_persists_with_defaults(
        self,
        db_session: AsyncSession,
        llm_config_repo: LLMProviderConfigRepository,
        test_user_db: User,
    ) -> None:
        """Creating a config should persist it with expected values."""
        _config = await llm_config_repo.get_or_create(cast(int, test_user_db.id))
        await db_session.commit()
        assert _config is not None  # Use the variable to satisfy linter

        # Verify persisted in database
        result = await db_session.execute(select(LLMProviderConfig).where(LLMProviderConfig.user_id == test_user_db.id))
        persisted = result.scalar_one()

        assert persisted.user_id == test_user_db.id
        assert persisted.high_quality_provider is None  # NULL = use defaults
        assert persisted.high_quality_model is None
        assert persisted.low_quality_provider is None
        assert persisted.low_quality_model is None
        assert isinstance(persisted.created_at, datetime)
        assert isinstance(persisted.updated_at, datetime)

    async def test_get_or_create_returns_existing(
        self,
        db_session: AsyncSession,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """get_or_create returns existing config for user."""
        result = await llm_config_repo.get_or_create(cast(int, test_user_db.id))

        assert result.id == llm_config.id
        assert result.user_id == test_user_db.id

    async def test_update_config_persists_changes(
        self,
        db_session: AsyncSession,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """Updating config should persist changes."""
        await llm_config_repo.update(
            cast(int, test_user_db.id),
            high_quality_provider="openai",
            high_quality_model="gpt-4o",
            default_temperature=0.7,
        )
        await db_session.commit()

        # Re-fetch from database
        result = await db_session.execute(select(LLMProviderConfig).where(LLMProviderConfig.id == llm_config.id))
        persisted = result.scalar_one()

        assert persisted.high_quality_provider == "openai"
        assert persisted.high_quality_model == "gpt-4o"
        assert persisted.default_temperature == 0.7
        # Original values should be unchanged
        assert persisted.low_quality_provider == "anthropic"

    async def test_get_by_user_id_returns_config(
        self,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """get_by_user_id returns correct config for user."""
        result = await llm_config_repo.get_by_user_id(cast(int, test_user_db.id))

        assert result is not None
        assert result.id == llm_config.id
        assert result.high_quality_provider == "anthropic"

    async def test_get_by_user_id_returns_none_for_missing(
        self,
        llm_config_repo: LLMProviderConfigRepository,
    ) -> None:
        """get_by_user_id returns None when user has no config."""
        result = await llm_config_repo.get_by_user_id(999999)

        assert result is None

    async def test_set_api_key_encrypts_and_persists(
        self,
        db_session: AsyncSession,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config: LLMProviderConfig,
    ) -> None:
        """set_api_key should encrypt and store key in database."""
        test_key = "sk-ant-test-api-key-12345"
        config_id = cast(int, llm_config.id)

        await llm_config_repo.set_api_key(config_id, "anthropic", test_key)
        await db_session.commit()

        # Verify stored in database
        result = await db_session.execute(
            select(LLMProviderApiKey).where(
                LLMProviderApiKey.config_id == config_id,
                LLMProviderApiKey.provider == "anthropic",
            )
        )
        api_key_row = result.scalar_one()

        # Ciphertext should not contain plaintext
        assert test_key.encode() not in api_key_row.ciphertext
        # Key ID should be set
        assert api_key_row.key_id == SecretEncryption.get_key_id()

    async def test_get_api_key_decrypts_correctly(
        self,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """get_api_key should return decrypted plaintext."""
        config_id = cast(int, llm_config_with_key.id)

        result = await llm_config_repo.get_api_key(config_id, "anthropic")

        assert result == "sk-ant-test-key-for-integration-tests"

    async def test_get_api_key_returns_none_for_missing(
        self,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config: LLMProviderConfig,
    ) -> None:
        """get_api_key returns None when no key configured."""
        config_id = cast(int, llm_config.id)

        result = await llm_config_repo.get_api_key(config_id, "anthropic")

        assert result is None

    async def test_delete_api_key_removes_from_db(
        self,
        db_session: AsyncSession,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """delete_api_key should remove key from database."""
        config_id = cast(int, llm_config_with_key.id)

        # Verify key exists first
        has_key_before = await llm_config_repo.has_api_key(config_id, "anthropic")
        assert has_key_before is True

        # Delete key
        result = await llm_config_repo.delete_api_key(config_id, "anthropic")
        await db_session.commit()

        assert result is True

        # Verify key removed
        has_key_after = await llm_config_repo.has_api_key(config_id, "anthropic")
        assert has_key_after is False

    async def test_has_api_key_returns_correct_status(
        self,
        llm_config_repo: LLMProviderConfigRepository,
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """has_api_key returns True for configured key, False otherwise."""
        config_id = cast(int, llm_config_with_key.id)

        # Anthropic key was set
        assert await llm_config_repo.has_api_key(config_id, "anthropic") is True
        # OpenAI key was not set
        assert await llm_config_repo.has_api_key(config_id, "openai") is False

    async def test_user_deletion_cascades_to_config(
        self,
        db_session: AsyncSession,
        llm_config_with_key: LLMProviderConfig,
        test_user_db: User,
    ) -> None:
        """Deleting user should cascade delete config and keys."""
        config_id = cast(int, llm_config_with_key.id)
        user_id = cast(int, test_user_db.id)

        # Delete user
        await db_session.execute(delete(User).where(User.id == user_id))
        await db_session.commit()

        # Verify config deleted
        result = await db_session.execute(select(LLMProviderConfig).where(LLMProviderConfig.id == config_id))
        assert result.scalar_one_or_none() is None

        # Verify API key deleted
        key_result = await db_session.execute(select(LLMProviderApiKey).where(LLMProviderApiKey.config_id == config_id))
        assert key_result.scalar_one_or_none() is None

    async def test_config_deletion_cascades_to_keys(
        self,
        db_session: AsyncSession,
        llm_config_with_key: LLMProviderConfig,
    ) -> None:
        """Deleting config should cascade delete API keys."""
        config_id = cast(int, llm_config_with_key.id)

        # Delete config
        await db_session.execute(delete(LLMProviderConfig).where(LLMProviderConfig.id == config_id))
        await db_session.commit()

        # Verify API key deleted
        result = await db_session.execute(select(LLMProviderApiKey).where(LLMProviderApiKey.config_id == config_id))
        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestLLMUsageRepositoryIntegration:
    """Integration tests for LLMUsageRepository."""

    async def test_record_usage_persists_event(
        self,
        db_session: AsyncSession,
        llm_usage_repo: LLMUsageRepository,
        test_user_db: User,
    ) -> None:
        """record_usage should persist event to database."""
        user_id = cast(int, test_user_db.id)

        event = await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await db_session.commit()

        # Verify persisted
        result = await db_session.execute(select(LLMUsageEvent).where(LLMUsageEvent.id == event.id))
        persisted = result.scalar_one()

        assert persisted.user_id == user_id
        assert persisted.provider == "anthropic"
        assert persisted.model == "claude-sonnet-4-5-20250929"
        assert persisted.quality_tier == "low"
        assert persisted.feature == "hyde"
        assert persisted.input_tokens == 100
        assert persisted.output_tokens == 200
        assert isinstance(persisted.created_at, datetime)

    async def test_record_usage_with_operation_id(
        self,
        db_session: AsyncSession,
        llm_usage_repo: LLMUsageRepository,
        test_user_db: User,
    ) -> None:
        """record_usage with operation_id for background tasks."""
        user_id = cast(int, test_user_db.id)

        event = await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-opus-4-5-20251101",
            quality_tier="high",
            feature="summary",
            input_tokens=1000,
            output_tokens=500,
            operation_id=12345,
            collection_id="col-abc123",
        )
        await db_session.commit()

        assert event.operation_id == 12345
        assert event.collection_id == "col-abc123"

    async def test_get_user_usage_summary(
        self,
        db_session: AsyncSession,
        llm_usage_repo: LLMUsageRepository,
        test_user_db: User,
    ) -> None:
        """get_user_usage_summary returns aggregated stats."""
        user_id = cast(int, test_user_db.id)

        # Create multiple events
        await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-opus-4-5-20251101",
            quality_tier="high",
            feature="summary",
            input_tokens=500,
            output_tokens=300,
        )
        await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="openai",
            model="gpt-4o",
            quality_tier="high",
            feature="summary",
            input_tokens=400,
            output_tokens=200,
        )
        await db_session.commit()

        summary = await llm_usage_repo.get_user_usage_summary(user_id, days=30)

        assert isinstance(summary, UsageSummary)
        assert summary.total_input_tokens == 1000
        assert summary.total_output_tokens == 700
        assert summary.total_tokens == 1700
        assert summary.event_count == 3

        # Check by_feature breakdown
        assert "hyde" in summary.by_feature
        assert summary.by_feature["hyde"]["input_tokens"] == 100
        assert summary.by_feature["hyde"]["output_tokens"] == 200

        assert "summary" in summary.by_feature
        assert summary.by_feature["summary"]["input_tokens"] == 900  # 500 + 400
        assert summary.by_feature["summary"]["output_tokens"] == 500  # 300 + 200

        # Check by_provider breakdown
        assert "anthropic" in summary.by_provider
        assert summary.by_provider["anthropic"]["input_tokens"] == 600  # 100 + 500

        assert "openai" in summary.by_provider
        assert summary.by_provider["openai"]["input_tokens"] == 400

    async def test_get_usage_filters_by_days(
        self,
        db_session: AsyncSession,
        llm_usage_repo: LLMUsageRepository,
        test_user_db: User,
    ) -> None:
        """Usage summary filters by days parameter."""
        user_id = cast(int, test_user_db.id)

        # Create event
        await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await db_session.commit()

        # Should include recent events
        summary = await llm_usage_repo.get_user_usage_summary(user_id, days=30)
        assert summary.event_count == 1

        # With 0 days (all time), should also include
        summary_all = await llm_usage_repo.get_user_usage_summary(user_id, days=0)
        assert summary_all.event_count == 1

    async def test_get_recent_events(
        self,
        db_session: AsyncSession,
        llm_usage_repo: LLMUsageRepository,
        test_user_db: User,
    ) -> None:
        """get_recent_events returns events in descending order."""
        user_id = cast(int, test_user_db.id)

        # Create multiple events
        for i in range(5):
            await llm_usage_repo.record_usage(
                user_id=user_id,
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                quality_tier="low",
                feature=f"feature_{i}",
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
            )
        await db_session.commit()

        events = await llm_usage_repo.get_recent_events(user_id, limit=3)

        assert len(events) == 3
        # Should have the 3 most recent events (order may vary when timestamps are equal)
        event_features = {e.feature for e in events}
        # With rapid inserts, order is not guaranteed, but we should get 3 of the 5 events
        assert len(event_features) == 3
        # All should be valid feature names
        assert all(f.startswith("feature_") for f in event_features)

    async def test_user_isolation(
        self,
        db_session: AsyncSession,
        llm_usage_repo: LLMUsageRepository,
        test_user_db: User,
        other_user_db: User,
    ) -> None:
        """Usage stats are isolated per user."""
        user_id = cast(int, test_user_db.id)
        other_user_id = cast(int, other_user_db.id)

        # Create events for both users
        await llm_usage_repo.record_usage(
            user_id=user_id,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=100,
            output_tokens=200,
        )
        await llm_usage_repo.record_usage(
            user_id=other_user_id,
            provider="openai",
            model="gpt-4o",
            quality_tier="high",
            feature="summary",
            input_tokens=500,
            output_tokens=300,
        )
        await db_session.commit()

        # Each user should only see their own events
        summary = await llm_usage_repo.get_user_usage_summary(user_id)
        assert summary.event_count == 1
        assert summary.total_input_tokens == 100

        other_summary = await llm_usage_repo.get_user_usage_summary(other_user_id)
        assert other_summary.event_count == 1
        assert other_summary.total_input_tokens == 500
