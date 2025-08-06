#!/usr/bin/env python3
"""
Unit tests for startup_tasks module.

This module tests the application startup tasks including
initialization of default data.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.webui.startup_tasks import (
    ensure_default_chunking_strategies,
    ensure_default_data,
)


class TestStartupTasks:
    """Tests for startup tasks."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture()
    def mock_chunking_service(self) -> MagicMock:
        """Create a mock chunking strategy service."""
        service = MagicMock()
        service.ensure_default_strategies = AsyncMock(return_value=3)
        return service

    @pytest.mark.asyncio()
    async def test_ensure_default_data_success(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test successful execution of ensure_default_data."""
        with (
            patch("packages.webui.startup_tasks.get_db") as mock_get_db,
            patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class,
        ):

            # Mock the async generator
            async def mock_db_generator():
                yield mock_session

            mock_get_db.return_value = mock_db_generator()
            mock_service_class.return_value = mock_chunking_service

            with caplog.at_level(logging.INFO):
                await ensure_default_data()

            # Verify the service was called
            mock_service_class.assert_called_once_with(mock_session)
            mock_chunking_service.ensure_default_strategies.assert_called_once()

            # Check log messages
            assert "Running startup tasks to ensure default data..." in caplog.text
            assert "Startup tasks completed" in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_chunking_strategies_creates_new(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test creating new chunking strategies when none exist."""
        mock_chunking_service.ensure_default_strategies.return_value = 3

        with patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class:
            mock_service_class.return_value = mock_chunking_service

            with caplog.at_level(logging.INFO):
                await ensure_default_chunking_strategies(mock_session)

            mock_service_class.assert_called_once_with(mock_session)
            mock_chunking_service.ensure_default_strategies.assert_called_once()

            # Check that it logged the creation
            assert "Created 3 default chunking strategies" in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_chunking_strategies_already_exist(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test when all default strategies already exist."""
        mock_chunking_service.ensure_default_strategies.return_value = 0

        with patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class:
            mock_service_class.return_value = mock_chunking_service

            with caplog.at_level(logging.DEBUG):
                await ensure_default_chunking_strategies(mock_session)

            mock_service_class.assert_called_once_with(mock_session)
            mock_chunking_service.ensure_default_strategies.assert_called_once()

            # Check that it logged at debug level
            assert "All default chunking strategies already exist" in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_chunking_strategies_handles_error(
        self,
        mock_session: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test error handling when creating default strategies fails."""
        error_message = "Database connection failed"

        with patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class:
            mock_service_class.side_effect = Exception(error_message)

            with caplog.at_level(logging.ERROR):
                # Should not raise exception, just log it
                await ensure_default_chunking_strategies(mock_session)

            # Check error was logged
            assert "Error ensuring default chunking strategies" in caplog.text
            assert error_message in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_chunking_strategies_service_error(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test when the service method raises an error."""
        error_message = "Failed to create strategy"
        mock_chunking_service.ensure_default_strategies.side_effect = Exception(error_message)

        with patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class:
            mock_service_class.return_value = mock_chunking_service

            with caplog.at_level(logging.ERROR):
                # Should not raise exception, just log it
                await ensure_default_chunking_strategies(mock_session)

            # Check error was logged
            assert "Error ensuring default chunking strategies" in caplog.text
            assert error_message in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_data_continues_on_error(
        self,
        mock_session: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that ensure_default_data continues even if chunking strategies fail."""
        with (
            patch("packages.webui.startup_tasks.get_db") as mock_get_db,
            patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class,
        ):

            # Mock the async generator
            async def mock_db_generator():
                yield mock_session

            mock_get_db.return_value = mock_db_generator()
            mock_service_class.side_effect = Exception("Service initialization failed")

            with caplog.at_level(logging.INFO):
                # Should complete without raising
                await ensure_default_data()

            # Check that startup still completed
            assert "Running startup tasks to ensure default data..." in caplog.text
            assert "Startup tasks completed" in caplog.text
            assert "Error ensuring default chunking strategies" in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_data_with_partial_success(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test when some strategies are created successfully."""
        mock_chunking_service.ensure_default_strategies.return_value = 1

        with (
            patch("packages.webui.startup_tasks.get_db") as mock_get_db,
            patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class,
        ):

            async def mock_db_generator():
                yield mock_session

            mock_get_db.return_value = mock_db_generator()
            mock_service_class.return_value = mock_chunking_service

            with caplog.at_level(logging.INFO):
                await ensure_default_data()

            # Check that it logged the partial creation
            assert "Created 1 default chunking strategies" in caplog.text

    @pytest.mark.asyncio()
    async def test_ensure_default_data_database_unavailable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test when database is unavailable."""

        # Mock the async generator to raise an exception
        async def mock_db_generator():
            raise Exception("Database connection refused")
            yield  # This won't be reached

        with (
            patch("packages.webui.startup_tasks.get_db") as mock_get_db,
            caplog.at_level(logging.ERROR),
        ):
            mock_get_db.return_value = mock_db_generator()

            # Should handle the error gracefully
            with pytest.raises(Exception, match="Database connection refused"):
                await ensure_default_data()

    @pytest.mark.asyncio()
    async def test_ensure_default_chunking_strategies_idempotent(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
    ) -> None:
        """Test that ensure_default_chunking_strategies is idempotent."""
        # First call creates strategies
        mock_chunking_service.ensure_default_strategies.return_value = 3

        with patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class:
            mock_service_class.return_value = mock_chunking_service

            await ensure_default_chunking_strategies(mock_session)

            # Second call finds them already exist
            mock_chunking_service.ensure_default_strategies.return_value = 0

            await ensure_default_chunking_strategies(mock_session)

            # Service should be called twice
            assert mock_chunking_service.ensure_default_strategies.call_count == 2

    @pytest.mark.asyncio()
    async def test_ensure_default_data_logs_appropriately(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that logging levels are appropriate."""
        mock_chunking_service.ensure_default_strategies.return_value = 0

        with (
            patch("packages.webui.startup_tasks.get_db") as mock_get_db,
            patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class,
        ):

            async def mock_db_generator():
                yield mock_session

            mock_get_db.return_value = mock_db_generator()
            mock_service_class.return_value = mock_chunking_service

            # Set different log levels and verify appropriate messages
            with caplog.at_level(logging.DEBUG):
                await ensure_default_data()

            # INFO level messages
            info_records = [r for r in caplog.records if r.levelname == "INFO"]
            assert len(info_records) >= 2  # Start and complete messages

            # DEBUG level messages when nothing created
            debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
            assert len(debug_records) >= 1  # "Already exist" message

    @pytest.mark.asyncio()
    async def test_ensure_default_chunking_strategies_with_none_session(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test behavior when session is None."""
        with caplog.at_level(logging.ERROR):
            # Should handle None session gracefully without raising
            await ensure_default_chunking_strategies(None)

            # Check that error was logged
            assert "Error ensuring default chunking strategies" in caplog.text

    @pytest.mark.asyncio()
    async def test_startup_tasks_concurrent_execution(
        self,
        mock_session: AsyncMock,
        mock_chunking_service: MagicMock,
    ) -> None:
        """Test that startup tasks can handle concurrent execution."""
        mock_chunking_service.ensure_default_strategies.return_value = 3

        with (
            patch("packages.webui.startup_tasks.get_db") as mock_get_db,
            patch("packages.webui.startup_tasks.ChunkingStrategyService") as mock_service_class,
        ):

            # Create separate generators for each call
            call_count = 0

            def create_db_generator():
                nonlocal call_count
                call_count += 1

                async def mock_db_generator():
                    yield mock_session

                return mock_db_generator()

            mock_get_db.side_effect = create_db_generator
            mock_service_class.return_value = mock_chunking_service

            # Run multiple concurrent executions
            import asyncio

            tasks = [ensure_default_data() for _ in range(3)]
            await asyncio.gather(*tasks)

            # Should handle concurrent calls without issues
            assert call_count == 3
            assert mock_chunking_service.ensure_default_strategies.call_count == 3
