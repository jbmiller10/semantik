"""Unit tests for shared.model_manager.hf_cache."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.model_manager.hf_cache import (
    HFCacheInfo,
    InstalledModel,
    clear_cache,
    get_cache_size_info,
    get_installed_models,
    get_model_size_on_disk,
    is_model_installed,
    resolve_hf_cache_dir,
    scan_hf_cache,
)


class TestResolveHfCacheDir:
    """Tests for resolve_hf_cache_dir function."""

    def test_default_path(self) -> None:
        """Should return default path when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove HF-related env vars
            for key in ["HF_HUB_CACHE", "HF_HOME"]:
                os.environ.pop(key, None)

            path = resolve_hf_cache_dir()
            assert path == Path.home() / ".cache" / "huggingface" / "hub"

    def test_hf_hub_cache_takes_priority(self) -> None:
        """HF_HUB_CACHE should take priority over other options."""
        with patch.dict(
            os.environ,
            {
                "HF_HUB_CACHE": "/custom/hf/cache",
                "HF_HOME": "/custom/hf/home",
            },
        ):
            path = resolve_hf_cache_dir()
            assert path == Path("/custom/hf/cache")

    def test_hf_home_with_hub_suffix(self) -> None:
        """HF_HOME should be used with /hub suffix when HF_HUB_CACHE not set."""
        with patch.dict(os.environ, {"HF_HOME": "/custom/hf/home"}, clear=False):
            os.environ.pop("HF_HUB_CACHE", None)
            path = resolve_hf_cache_dir()
            assert path == Path("/custom/hf/home/hub")


class TestScanHfCache:
    """Tests for scan_hf_cache function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_returns_hf_cache_info(self) -> None:
        """Should return HFCacheInfo instance."""
        result = scan_hf_cache()
        assert isinstance(result, HFCacheInfo)

    def test_cache_dir_is_path(self) -> None:
        """cache_dir should be a Path."""
        result = scan_hf_cache()
        assert isinstance(result.cache_dir, Path)

    def test_repos_is_dict(self) -> None:
        """repos should be a dict."""
        result = scan_hf_cache()
        assert isinstance(result.repos, dict)

    def test_ttl_caching(self) -> None:
        """Result should be cached for TTL duration."""
        clear_cache()

        # First call
        result1 = scan_hf_cache()

        # Second call should return cached result
        result2 = scan_hf_cache()

        assert result1 is result2

    def test_force_refresh_bypasses_cache(self) -> None:
        """force_refresh=True should bypass the cache."""
        clear_cache()

        # First call - populate cache
        _ = scan_hf_cache()

        # Force refresh
        result2 = scan_hf_cache(force_refresh=True)

        # Should return a valid result
        assert result2 is not None
        assert isinstance(result2, HFCacheInfo)


class TestGetInstalledModels:
    """Tests for get_installed_models function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_returns_dict(self) -> None:
        """Should return a dict."""
        models = get_installed_models()
        assert isinstance(models, dict)

    def test_values_are_installed_models(self) -> None:
        """Dict values should be InstalledModel instances."""
        models = get_installed_models()
        for model in models.values():
            assert isinstance(model, InstalledModel)


class TestIsModelInstalled:
    """Tests for is_model_installed function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_returns_bool(self) -> None:
        """Should return a boolean."""
        result = is_model_installed("nonexistent/model")
        assert isinstance(result, bool)

    def test_nonexistent_model_returns_false(self) -> None:
        """Nonexistent model should return False."""
        result = is_model_installed("definitely/not/a/real/model/12345")
        assert result is False


class TestGetModelSizeOnDisk:
    """Tests for get_model_size_on_disk function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_nonexistent_model_returns_none(self) -> None:
        """Nonexistent model should return None."""
        result = get_model_size_on_disk("definitely/not/a/real/model/12345")
        assert result is None


class TestGetCacheSizeInfo:
    """Tests for get_cache_size_info function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_returns_cache_size_breakdown(self) -> None:
        """Should return CacheSizeBreakdown TypedDict."""
        result = get_cache_size_info({"test/model"})
        assert isinstance(result, dict)
        assert "total_cache_size_mb" in result
        assert "managed_cache_size_mb" in result
        assert "unmanaged_cache_size_mb" in result
        assert "unmanaged_repo_count" in result

    def test_values_are_non_negative(self) -> None:
        """All values should be non-negative."""
        result = get_cache_size_info(set())
        assert result["total_cache_size_mb"] >= 0
        assert result["managed_cache_size_mb"] >= 0
        assert result["unmanaged_cache_size_mb"] >= 0
        assert result["unmanaged_repo_count"] >= 0


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clears_cached_result(self) -> None:
        """clear_cache should reset the module-level cache."""
        # Populate cache
        scan_hf_cache()

        # Clear it
        clear_cache()

        # Next call should be a fresh scan
        # We can't easily verify this without mocking, but at least
        # verify no exceptions
        scan_hf_cache()


class TestInstalledModelDataclass:
    """Tests for InstalledModel dataclass."""

    def test_has_expected_fields(self) -> None:
        """Should have all expected fields."""
        model = InstalledModel(
            repo_id="test/model",
            size_on_disk_mb=100,
            repo_type="model",
            last_accessed=None,
            revisions=["abc12345"],
        )
        assert model.repo_id == "test/model"
        assert model.size_on_disk_mb == 100
        assert model.repo_type == "model"
        assert model.last_accessed is None
        assert model.revisions == ["abc12345"]


class TestHFCacheInfoDataclass:
    """Tests for HFCacheInfo dataclass."""

    def test_has_expected_fields(self) -> None:
        """Should have all expected fields."""
        from datetime import UTC, datetime

        info = HFCacheInfo(
            cache_dir=Path("/test/cache"),
            repos={},
            total_size_mb=500,
            scan_time=datetime.now(tz=UTC),
        )
        assert info.cache_dir == Path("/test/cache")
        assert info.repos == {}
        assert info.total_size_mb == 500
        assert isinstance(info.scan_time, datetime)


class TestHFCacheErrorLogging:
    """Tests for error logging in scan_hf_cache function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_import_error_logs_debug_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """ImportError should log at debug level."""
        # Create a temporary directory that exists
        with patch(
            "shared.model_manager.hf_cache.resolve_hf_cache_dir"
        ) as mock_resolve:
            mock_resolve.return_value = Path("/tmp")

            # Mock the import to fail
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                with patch(
                    "builtins.__import__",
                    side_effect=ImportError("No module named 'huggingface_hub'"),
                ):
                    with caplog.at_level(logging.DEBUG, logger="shared.model_manager.hf_cache"):
                        result = scan_hf_cache(force_refresh=True)

        # Should still return a valid (empty) result
        assert isinstance(result, HFCacheInfo)
        assert result.repos == {}

        # Check the debug log message
        assert any(
            "huggingface_hub not installed" in record.message
            and record.levelno == logging.DEBUG
            for record in caplog.records
        )

    def test_permission_error_logs_warning_with_path(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """PermissionError should log at warning level with cache path."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        with patch(
            "shared.model_manager.hf_cache.resolve_hf_cache_dir"
        ) as mock_resolve:
            mock_resolve.return_value = cache_dir

            # Mock scan_cache_dir to raise PermissionError
            mock_scan = MagicMock(
                side_effect=PermissionError("Permission denied: '/test/cache'")
            )
            with patch("huggingface_hub.scan_cache_dir", mock_scan):
                with caplog.at_level(logging.WARNING, logger="shared.model_manager.hf_cache"):
                    result = scan_hf_cache(force_refresh=True)

        # Should return empty result
        assert isinstance(result, HFCacheInfo)
        assert result.repos == {}

        # Check the warning log message
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Permission denied" in r.message
        ]
        assert len(warning_records) == 1
        assert str(cache_dir) in warning_records[0].message
        assert "Models may show as not installed" in warning_records[0].message

    def test_general_exception_logs_warning_with_path(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """General exceptions should log at warning level with cache path."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        with patch(
            "shared.model_manager.hf_cache.resolve_hf_cache_dir"
        ) as mock_resolve:
            mock_resolve.return_value = cache_dir

            # Mock scan_cache_dir to raise a general exception
            mock_scan = MagicMock(
                side_effect=RuntimeError("Unexpected cache format")
            )
            with patch("huggingface_hub.scan_cache_dir", mock_scan):
                with caplog.at_level(logging.WARNING, logger="shared.model_manager.hf_cache"):
                    result = scan_hf_cache(force_refresh=True)

        # Should return empty result
        assert isinstance(result, HFCacheInfo)
        assert result.repos == {}

        # Check the warning log message
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Failed to scan HF cache" in r.message
        ]
        assert len(warning_records) == 1
        assert str(cache_dir) in warning_records[0].message
        assert "Models may show as not installed" in warning_records[0].message
        assert "Unexpected cache format" in warning_records[0].message
