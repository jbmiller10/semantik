"""Repository implementation for LLM usage tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError
from shared.database.models import LLMUsageEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageSummary:
    """Summary of LLM token usage for a user.

    Attributes:
        total_input_tokens: Total input tokens across all features
        total_output_tokens: Total output tokens across all features
        total_tokens: Combined input + output tokens
        by_feature: Breakdown by feature (e.g., {'hyde': {...}, 'summary': {...}})
        by_provider: Breakdown by provider (e.g., {'anthropic': {...}, 'openai': {...}})
        event_count: Total number of usage events
        period_days: Number of days covered by this summary
    """

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    by_feature: dict[str, dict[str, int]]
    by_provider: dict[str, dict[str, int]]
    event_count: int
    period_days: int


class LLMUsageRepository:
    """Repository for LLM usage event tracking.

    This repository manages token usage tracking for LLM requests,
    supporting both interactive requests (HyDE search) and background
    operations (summarization).

    Example:
        ```python
        repo = LLMUsageRepository(session)

        # Record usage after an LLM call
        event = await repo.record_usage(
            user_id=123,
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            quality_tier="low",
            feature="hyde",
            input_tokens=150,
            output_tokens=300,
        )

        # Get usage summary
        summary = await repo.get_user_usage_summary(user_id=123, days=30)
        print(f"Total tokens: {summary.total_tokens}")
        ```
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def record_usage(
        self,
        user_id: int,
        provider: str,
        model: str,
        quality_tier: str,
        feature: str,
        input_tokens: int,
        output_tokens: int,
        *,
        operation_id: int | None = None,
        collection_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> LLMUsageEvent:
        """Record an LLM usage event.

        Args:
            user_id: The user's ID
            provider: Provider name ('anthropic', 'openai')
            model: Model identifier
            quality_tier: Quality tier used ('high', 'low')
            feature: Feature name ('hyde', 'summary', 'extraction')
            input_tokens: Number of input tokens (provider-reported)
            output_tokens: Number of output tokens (provider-reported)
            operation_id: Optional operation ID (NULL for interactive requests)
            collection_id: Optional collection ID for context
            request_metadata: Optional metadata dict

        Returns:
            Created LLMUsageEvent instance

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            event = LLMUsageEvent(
                user_id=user_id,
                provider=provider,
                model=model,
                quality_tier=quality_tier,
                feature=feature,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation_id=operation_id,
                collection_id=collection_id,
                request_metadata=request_metadata,
            )

            self.session.add(event)
            await self.session.flush()

            logger.debug(
                f"Recorded LLM usage: feature={feature}, provider={provider}, " f"tokens={input_tokens}+{output_tokens}"
            )

            return event

        except Exception as e:
            logger.error("Failed to record LLM usage: %s", e, exc_info=True)
            raise DatabaseOperationError("record", "LLMUsageEvent", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_user_usage_summary(
        self,
        user_id: int,
        days: int = 30,
    ) -> UsageSummary:
        """Get aggregated usage summary for a user.

        Args:
            user_id: The user's ID
            days: Number of days to include (default 30, 0 for all-time)

        Returns:
            UsageSummary with totals and breakdowns

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Build base query with user filter
            base_filter = [LLMUsageEvent.user_id == user_id]

            # Add date filter if needed
            if days > 0:
                cutoff = datetime.now(UTC) - timedelta(days=days)
                base_filter.append(LLMUsageEvent.created_at >= cutoff)

            # Get total tokens
            totals_query = select(
                func.coalesce(func.sum(LLMUsageEvent.input_tokens), 0).label("input"),
                func.coalesce(func.sum(LLMUsageEvent.output_tokens), 0).label("output"),
                func.count(LLMUsageEvent.id).label("cnt"),
            ).where(*base_filter)

            totals_result = await self.session.execute(totals_query)
            totals_row = totals_result.one()

            total_input = int(totals_row.input)
            total_output = int(totals_row.output)
            event_count = int(totals_row.cnt)

            # Get breakdown by feature
            feature_query = (
                select(
                    LLMUsageEvent.feature,
                    func.sum(LLMUsageEvent.input_tokens).label("input"),
                    func.sum(LLMUsageEvent.output_tokens).label("output"),
                    func.count(LLMUsageEvent.id).label("cnt"),
                )
                .where(*base_filter)
                .group_by(LLMUsageEvent.feature)
            )

            feature_result = await self.session.execute(feature_query)
            by_feature: dict[str, dict[str, int]] = {}
            for row in feature_result.all():
                input_val = int(row.input or 0)
                output_val = int(row.output or 0)
                by_feature[row.feature] = {
                    "input_tokens": input_val,
                    "output_tokens": output_val,
                    "total_tokens": input_val + output_val,
                    "count": int(row.cnt or 0),
                }

            # Get breakdown by provider
            provider_query = (
                select(
                    LLMUsageEvent.provider,
                    func.sum(LLMUsageEvent.input_tokens).label("input"),
                    func.sum(LLMUsageEvent.output_tokens).label("output"),
                    func.count(LLMUsageEvent.id).label("cnt"),
                )
                .where(*base_filter)
                .group_by(LLMUsageEvent.provider)
            )

            provider_result = await self.session.execute(provider_query)
            by_provider: dict[str, dict[str, int]] = {}
            for row in provider_result.all():
                input_val = int(row.input or 0)
                output_val = int(row.output or 0)
                by_provider[row.provider] = {
                    "input_tokens": input_val,
                    "output_tokens": output_val,
                    "total_tokens": input_val + output_val,
                    "count": int(row.cnt or 0),
                }

            return UsageSummary(
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_tokens=total_input + total_output,
                by_feature=by_feature,
                by_provider=by_provider,
                event_count=event_count,
                period_days=days,
            )

        except Exception as e:
            logger.error("Failed to get usage summary for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get_summary", "LLMUsageEvent", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_usage_by_feature(
        self,
        user_id: int,
        feature: str,
        days: int = 30,
        limit: int = 100,
    ) -> list[LLMUsageEvent]:
        """Get usage events for a specific feature.

        Args:
            user_id: The user's ID
            feature: Feature name to filter by
            days: Number of days to include (default 30, 0 for all-time)
            limit: Maximum number of events to return

        Returns:
            List of LLMUsageEvent instances, most recent first

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            query = select(LLMUsageEvent).where(
                LLMUsageEvent.user_id == user_id,
                LLMUsageEvent.feature == feature,
            )

            if days > 0:
                cutoff = datetime.now(UTC) - timedelta(days=days)
                query = query.where(LLMUsageEvent.created_at >= cutoff)

            query = query.order_by(LLMUsageEvent.created_at.desc()).limit(limit)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error("Failed to get usage by feature %s: %s", feature, e, exc_info=True)
            raise DatabaseOperationError("get_by_feature", "LLMUsageEvent", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_total_tokens(
        self,
        user_id: int,
        days: int = 30,
    ) -> dict[str, int]:
        """Get total token counts for a user.

        Convenience method for simple token counting.

        Args:
            user_id: The user's ID
            days: Number of days to include (default 30, 0 for all-time)

        Returns:
            Dict with 'input_tokens', 'output_tokens', 'total_tokens'

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            base_filter = [LLMUsageEvent.user_id == user_id]

            if days > 0:
                cutoff = datetime.now(UTC) - timedelta(days=days)
                base_filter.append(LLMUsageEvent.created_at >= cutoff)

            query = select(
                func.coalesce(func.sum(LLMUsageEvent.input_tokens), 0).label("input"),
                func.coalesce(func.sum(LLMUsageEvent.output_tokens), 0).label("output"),
            ).where(*base_filter)

            result = await self.session.execute(query)
            row = result.one()

            input_tokens = int(row.input)
            output_tokens = int(row.output)

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        except Exception as e:
            logger.error("Failed to get total tokens for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get_total_tokens", "LLMUsageEvent", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_recent_events(
        self,
        user_id: int,
        limit: int = 10,
    ) -> list[LLMUsageEvent]:
        """Get most recent usage events for a user.

        Args:
            user_id: The user's ID
            limit: Maximum number of events to return

        Returns:
            List of LLMUsageEvent instances, most recent first

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            query = (
                select(LLMUsageEvent)
                .where(LLMUsageEvent.user_id == user_id)
                .order_by(LLMUsageEvent.created_at.desc())
                .limit(limit)
            )

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error("Failed to get recent events for user %s: %s", user_id, e, exc_info=True)
            raise DatabaseOperationError("get_recent", "LLMUsageEvent", str(e)) from e
