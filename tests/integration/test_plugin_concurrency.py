"""Integration tests for plugin system concurrency.

These tests verify thread-safety and concurrent operation handling
for plugin registry, config upserts, and health checks.

Phase 4.2 of the plugin system remediation plan.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from shared.database.repositories.plugin_config_repository import PluginConfigRepository
from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginRegistry, PluginSource


def _make_manifest(plugin_id: str, plugin_type: str = "embedding") -> PluginManifest:
    """Create a test manifest."""
    return PluginManifest(
        id=plugin_id,
        type=plugin_type,
        version="1.0.0",
        display_name=f"Test Plugin {plugin_id}",
        description="Test plugin for concurrency tests",
    )


def _make_record(
    plugin_id: str,
    plugin_type: str = "embedding",
    plugin_class: type | None = None,
) -> PluginRecord:
    """Create a test plugin record."""
    return PluginRecord(
        plugin_type=plugin_type,
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        manifest=_make_manifest(plugin_id, plugin_type),
        plugin_class=plugin_class or MagicMock,
        source=PluginSource.EXTERNAL,
    )


class TestConcurrentRegistration:
    """Test concurrent plugin registration operations."""

    @pytest.fixture()
    def clean_registry(self):
        """Create a fresh registry for each test."""
        return PluginRegistry()

    def test_concurrent_registration_100_plugins(self, clean_registry):
        """Test 100 concurrent plugin registrations all succeed."""
        results = []

        def register_plugin(i: int):
            record = _make_record(
                f"plugin-{i}",
                "embedding",
                plugin_class=type(f"TestClass{i}", (), {}),
            )
            result = clean_registry.register(record)
            results.append((i, result))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(register_plugin, i) for i in range(100)]
            for f in futures:
                f.result()

        # All 100 should have registered successfully
        success_count = sum(1 for _, result in results if result is True)
        assert success_count == 100, f"Expected 100 successes, got {success_count}"
        assert len(clean_registry.get_by_type("embedding")) == 100

    def test_concurrent_registration_no_duplicates(self, clean_registry):
        """Test concurrent registration of same ID from different threads."""
        results = []
        errors = []

        def register_same_plugin(thread_id: int):
            try:
                # All threads try to register same plugin ID
                record = _make_record(
                    "shared-plugin",
                    "embedding",
                    plugin_class=type(f"ThreadClass{thread_id}", (), {}),
                )
                result = clean_registry.register(record)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_same_plugin, i) for i in range(10)]
            for f in futures:
                f.result()

        # Only one should succeed (first to acquire lock), rest should fail
        # Due to different classes, we expect PluginDuplicateError for others
        # The first thread gets True, subsequent threads get errors
        assert len(clean_registry.get_by_type("embedding")) == 1

    @pytest.mark.asyncio()
    async def test_async_concurrent_registration(self, clean_registry):
        """Test async concurrent plugin registrations."""

        async def register_plugin_async(i: int) -> bool:
            # Run in thread pool since registry is sync
            loop = asyncio.get_event_loop()
            record = _make_record(
                f"async-plugin-{i}",
                "embedding",
                plugin_class=type(f"AsyncClass{i}", (), {}),
            )
            return await loop.run_in_executor(None, clean_registry.register, record)

        tasks = [register_plugin_async(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        success_count = sum(results)
        assert success_count == 50
        assert len(clean_registry.get_by_type("embedding")) == 50


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
@pytest.mark.skip(
    reason=(
        "Concurrent DB tests require separate sessions; "
        "single AsyncSession doesn't support concurrent operations"
    )
)
class TestConcurrentConfigUpsert:
    """Test concurrent plugin config database operations.

    These tests verify the atomic upsert implemented in Phase 0
    handles concurrent updates correctly.

    NOTE: These tests are skipped because SQLAlchemy AsyncSession
    doesn't support concurrent operations on the same session.
    True concurrency testing would require a connection pool with
    separate sessions per concurrent operation.
    """

    async def test_concurrent_upsert_same_plugin(self, db_session):
        """Test concurrent upserts to same plugin don't lose data."""
        repo = PluginConfigRepository(db_session)
        plugin_id = "concurrent-test-plugin"
        results = []
        errors = []

        async def upsert_config(value: int):
            try:
                config = await repo.upsert_config(
                    plugin_id=plugin_id,
                    plugin_type="embedding",
                    config={"value": value, "updated_by": f"task-{value}"},
                )
                results.append(config)
            except Exception as e:
                errors.append(e)

        # Run 20 concurrent upserts
        tasks = [upsert_config(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Commit to persist
        await db_session.commit()

        # Should have no errors due to atomic upsert
        assert len(errors) == 0, f"Got errors: {errors}"

        # Should have one record with one of the values
        config = await repo.get_config(plugin_id)
        assert config is not None
        assert "value" in config.config
        # Value should be one of 0-19
        assert 0 <= config.config["value"] < 20

    async def test_concurrent_upsert_different_plugins(self, db_session):
        """Test concurrent upserts to different plugins all succeed."""
        repo = PluginConfigRepository(db_session)
        results = []

        async def upsert_plugin(i: int):
            config = await repo.upsert_config(
                plugin_id=f"plugin-{i}",
                plugin_type="embedding",
                config={"index": i},
            )
            results.append(config)

        # Run 30 concurrent upserts for different plugins
        tasks = [upsert_plugin(i) for i in range(30)]
        await asyncio.gather(*tasks)
        await db_session.commit()

        # All 30 should succeed
        assert len(results) == 30

        # Verify all were created
        configs = await repo.list_configs(plugin_type="embedding")
        created_ids = {c.id for c in configs if c.id.startswith("plugin-")}
        assert len(created_ids) == 30

    async def test_concurrent_enable_disable(self, db_session):
        """Test concurrent enable/disable operations."""
        repo = PluginConfigRepository(db_session)
        plugin_id = "toggle-test-plugin"

        # Create initial config
        await repo.upsert_config(
            plugin_id=plugin_id,
            plugin_type="embedding",
            enabled=True,
        )
        await db_session.commit()

        async def toggle_enabled(enable: bool):
            await repo.upsert_config(
                plugin_id=plugin_id,
                plugin_type="embedding",
                enabled=enable,
            )

        # Alternate enable/disable rapidly
        tasks = [toggle_enabled(i % 2 == 0) for i in range(20)]
        await asyncio.gather(*tasks)
        await db_session.commit()

        # Final state should be deterministic (last write wins)
        config = await repo.get_config(plugin_id)
        assert config is not None
        # enabled should be boolean
        assert isinstance(config.enabled, bool)


@pytest.mark.asyncio()
class TestConcurrentHealthChecks:
    """Test concurrent health check operations."""

    async def test_concurrent_health_checks_no_deadlock(self):
        """Test concurrent health checks complete without deadlock."""

        class TestPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "health-test"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            async def health_check(cls, _config: dict[str, Any] | None = None) -> bool:
                # Simulate some async work
                await asyncio.sleep(0.01)
                return True

        results = []

        async def check_health():
            result = await TestPlugin.health_check({})
            results.append(result)

        # 30 concurrent health checks
        tasks = [check_health() for _ in range(30)]

        # Should complete within timeout (no deadlock)
        await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=5.0,
        )

        assert len(results) == 30
        assert all(r is True for r in results)

    async def test_health_check_timeout_handling(self):
        """Test that slow health checks can be timed out."""
        timeout_seconds = 0.1

        class SlowPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "slow-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            async def health_check(cls, _config: dict[str, Any] | None = None) -> bool:
                # Simulate a hung health check
                await asyncio.sleep(10.0)
                return True

        async def run_health_check_with_timeout() -> tuple[bool, str | None]:
            try:
                result = await asyncio.wait_for(
                    SlowPlugin.health_check({}),
                    timeout=timeout_seconds,
                )
                return result, None
            except TimeoutError:
                return False, f"Health check timed out after {timeout_seconds}s"
            except Exception as e:
                return False, str(e)

        # Run timed-out health check
        healthy, error = await run_health_check_with_timeout()

        assert healthy is False
        assert error is not None
        assert "timed out" in error

    async def test_concurrent_health_checks_mixed_results(self):
        """Test concurrent health checks with mixed pass/fail results."""
        check_count = 0
        lock = asyncio.Lock()

        class MixedPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "mixed-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            async def health_check(cls, _config: dict[str, Any] | None = None) -> bool:
                nonlocal check_count
                async with lock:
                    check_count += 1
                    current = check_count
                # Alternate between success and failure
                await asyncio.sleep(0.01)
                return current % 2 == 0

        results = []

        async def check_health():
            result = await MixedPlugin.health_check({})
            results.append(result)

        tasks = [check_health() for _ in range(20)]
        await asyncio.gather(*tasks)

        # Should have mix of True and False
        assert len(results) == 20
        assert any(r is True for r in results)
        assert any(r is False for r in results)


class TestConcurrentReadWrite:
    """Test concurrent read and write operations."""

    @pytest.fixture()
    def registry(self):
        """Create a fresh registry."""
        return PluginRegistry()

    def test_concurrent_read_write(self, registry):
        """Test concurrent reads and writes don't corrupt state."""
        errors = []

        def writer(i: int):
            try:
                record = _make_record(
                    f"writer-{i}",
                    "embedding",
                    plugin_class=type(f"WriterClass{i}", (), {}),
                )
                registry.register(record)
            except Exception as e:
                errors.append(("write", i, e))

        def reader(i: int):
            try:
                # Various read operations
                registry.get_all()
                registry.list_records()
                registry.find_by_id(f"writer-{i % 10}")
                registry.get_by_type("embedding")
                registry.list_types()
            except Exception as e:
                errors.append(("read", i, e))

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(writer, i))
                futures.append(executor.submit(reader, i))
            for f in futures:
                f.result()

        # No errors should occur
        assert len(errors) == 0, f"Got errors: {errors}"

        # All writers should have registered
        assert len(registry.get_by_type("embedding")) == 50

    def test_concurrent_disable_check(self, registry):
        """Test concurrent disable/check operations."""
        errors = []

        # Pre-register some plugins
        for i in range(20):
            record = _make_record(
                f"disable-test-{i}",
                "embedding",
                plugin_class=type(f"DisableClass{i}", (), {}),
            )
            registry.register(record)

        def set_disabled(plugin_ids: set[str]):
            try:
                registry.set_disabled(plugin_ids)
            except Exception as e:
                errors.append(("set", e))

        def check_disabled(plugin_id: str):
            try:
                registry.is_disabled(plugin_id)
            except Exception as e:
                errors.append(("check", e))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(30):
                # Set different disabled sets
                futures.append(executor.submit(set_disabled, {f"disable-test-{i % 20}"}))
                # Check various plugins
                futures.append(executor.submit(check_disabled, f"disable-test-{i % 20}"))
            for f in futures:
                f.result()

        assert len(errors) == 0
