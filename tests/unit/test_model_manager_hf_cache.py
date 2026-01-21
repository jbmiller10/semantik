"""Unit tests for shared.model_manager.hf_cache."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.model_manager.hf_cache import (
    HFCacheInfo,
    InstalledModel,
    _scan_single_cache_dir,
    clear_cache,
    get_cache_size_info,
    get_installed_models,
    get_model_size_on_disk,
    is_model_installed,
    resolve_hf_cache_dir,
    resolve_transformers_cache_dir,
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


class TestHfCacheRepoIdentityAndSizing:
    """Tests for HF cache repo identity and size semantics."""

    def setup_method(self) -> None:
        clear_cache()

    def test_repo_key_includes_repo_type(self, tmp_path: Path) -> None:
        """scan_hf_cache should not collide repo_id across repo types."""
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        # Two repos share repo_id but differ in type (valid on HF).
        repo_dataset = MagicMock()
        repo_dataset.repo_id = "org/repo"
        repo_dataset.repo_type = "dataset"
        repo_dataset.size_on_disk = 2 * 1024 * 1024
        repo_dataset.revisions = []

        repo_model = MagicMock()
        repo_model.repo_id = "org/repo"
        repo_model.repo_type = "model"
        repo_model.size_on_disk = 5 * 1024 * 1024
        repo_model.revisions = []

        scan_result = MagicMock()
        scan_result.size_on_disk = repo_dataset.size_on_disk + repo_model.size_on_disk
        scan_result.repos = [repo_dataset, repo_model]

        with patch("shared.model_manager.hf_cache.resolve_hf_cache_dir") as mock_resolve:
            mock_resolve.return_value = cache_dir
            with patch("huggingface_hub.scan_cache_dir", return_value=scan_result):
                info = scan_hf_cache(force_refresh=True)

                assert ("dataset", "org/repo") in info.repos
                assert ("model", "org/repo") in info.repos

                # Installed model checks should only consider repo_type=="model"
                models = get_installed_models()
                assert set(models.keys()) == {"org/repo"}
                assert models["org/repo"].repo_type == "model"

    def test_cache_size_semantics_reconcile_total(self, tmp_path: Path) -> None:
        """unmanaged_cache_size_mb should reconcile as (total - managed)."""
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        repo_dataset = MagicMock()
        repo_dataset.repo_id = "org/repo"
        repo_dataset.repo_type = "dataset"
        repo_dataset.size_on_disk = 2 * 1024 * 1024
        repo_dataset.revisions = []

        repo_model = MagicMock()
        repo_model.repo_id = "org/repo"
        repo_model.repo_type = "model"
        repo_model.size_on_disk = 5 * 1024 * 1024
        repo_model.revisions = []

        scan_result = MagicMock()
        scan_result.size_on_disk = repo_dataset.size_on_disk + repo_model.size_on_disk
        scan_result.repos = [repo_dataset, repo_model]

        with patch("shared.model_manager.hf_cache.resolve_hf_cache_dir") as mock_resolve:
            mock_resolve.return_value = cache_dir
            with patch("huggingface_hub.scan_cache_dir", return_value=scan_result):
                result = get_cache_size_info({"org/repo"})

        assert result["total_cache_size_mb"] == 7
        assert result["managed_cache_size_mb"] == 5
        assert result["unmanaged_cache_size_mb"] == 2
        assert result["unmanaged_repo_count"] == 1


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
        assert info.scan_error is None


class TestHFCacheErrorLogging:
    """Tests for error logging in scan_hf_cache function."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_import_error_logs_debug_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """ImportError should log at debug level."""
        # Create a temporary directory that exists
        with patch("shared.model_manager.hf_cache.resolve_hf_cache_dir") as mock_resolve:
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
        assert result.scan_error is not None

        # Check the debug log message
        assert any(
            "huggingface_hub not installed" in record.message and record.levelno == logging.DEBUG
            for record in caplog.records
        )

    def test_permission_error_logs_warning_with_path(self, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
        """PermissionError should log at warning level with cache path."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        with patch("shared.model_manager.hf_cache.resolve_hf_cache_dir") as mock_resolve:
            mock_resolve.return_value = cache_dir

            # Mock scan_cache_dir to raise PermissionError
            mock_scan = MagicMock(side_effect=PermissionError("Permission denied: '/test/cache'"))
            with patch("huggingface_hub.scan_cache_dir", mock_scan):
                with caplog.at_level(logging.WARNING, logger="shared.model_manager.hf_cache"):
                    result = scan_hf_cache(force_refresh=True)

        # Should return empty result
        assert isinstance(result, HFCacheInfo)
        assert result.repos == {}
        assert result.scan_error is not None

        # Check the warning log message
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "Permission denied" in r.message
        ]
        assert len(warning_records) == 1
        assert str(cache_dir) in warning_records[0].message
        assert "Models may show as not installed" in warning_records[0].message

    def test_general_exception_logs_warning_with_path(self, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
        """ValueError/OSError exceptions should log at warning level with cache path."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        with patch("shared.model_manager.hf_cache.resolve_hf_cache_dir") as mock_resolve:
            mock_resolve.return_value = cache_dir

            # Mock scan_cache_dir to raise a ValueError (e.g., unexpected cache format)
            mock_scan = MagicMock(side_effect=ValueError("Unexpected cache format"))
            with patch("huggingface_hub.scan_cache_dir", mock_scan):
                with caplog.at_level(logging.WARNING, logger="shared.model_manager.hf_cache"):
                    result = scan_hf_cache(force_refresh=True)

        # Should return empty result
        assert isinstance(result, HFCacheInfo)
        assert result.repos == {}
        assert result.scan_error is not None

        # Check the warning log message
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "Failed to scan cache" in r.message
        ]
        assert len(warning_records) == 1
        assert str(cache_dir) in warning_records[0].message
        assert "Models may show as not installed" in warning_records[0].message
        assert "Unexpected cache format" in warning_records[0].message

    def test_scan_failure_prefers_cached_repos(self, tmp_path: Path) -> None:
        """If a scan fails, return the previous cached repos instead of empty."""
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        repo_model = MagicMock()
        repo_model.repo_id = "org/repo"
        repo_model.repo_type = "model"
        repo_model.size_on_disk = 5 * 1024 * 1024
        repo_model.revisions = []

        scan_result = MagicMock()
        scan_result.size_on_disk = repo_model.size_on_disk
        scan_result.repos = [repo_model]

        with patch("shared.model_manager.hf_cache.resolve_hf_cache_dir") as mock_resolve:
            mock_resolve.return_value = cache_dir
            with patch("huggingface_hub.scan_cache_dir", return_value=scan_result):
                first = scan_hf_cache(force_refresh=True)

            with patch("huggingface_hub.scan_cache_dir", side_effect=PermissionError("Permission denied")):
                second = scan_hf_cache(force_refresh=True)

        assert ("model", "org/repo") in first.repos
        assert ("model", "org/repo") in second.repos
        assert second.repos == first.repos
        assert second.scan_error is not None


class TestResolveTransformersCacheDir:
    """Tests for resolve_transformers_cache_dir function."""

    def test_returns_none_when_not_set(self) -> None:
        """Should return None when TRANSFORMERS_CACHE is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRANSFORMERS_CACHE", None)
            result = resolve_transformers_cache_dir()
            assert result is None

    def test_returns_none_when_same_as_hub_path(self) -> None:
        """Should return None when TRANSFORMERS_CACHE equals HF_HUB_CACHE."""
        with patch.dict(
            os.environ,
            {
                "TRANSFORMERS_CACHE": "/app/.cache/huggingface/hub",
                "HF_HUB_CACHE": "/app/.cache/huggingface/hub",
            },
        ):
            result = resolve_transformers_cache_dir()
            assert result is None

    def test_returns_path_when_different_from_hub_path(self) -> None:
        """Should return path when TRANSFORMERS_CACHE differs from hub cache."""
        with patch.dict(
            os.environ,
            {
                "TRANSFORMERS_CACHE": "/app/.cache/huggingface",
                "HF_HUB_CACHE": "/app/.cache/huggingface/hub",
            },
        ):
            result = resolve_transformers_cache_dir()
            assert result == Path("/app/.cache/huggingface")


class TestScanSingleCacheDir:
    """Tests for _scan_single_cache_dir function."""

    def test_nonexistent_directory_returns_zero_no_error(self, tmp_path: Path) -> None:
        """Nonexistent directory should return (0, None)."""
        nonexistent = tmp_path / "nonexistent"
        repos: dict = {}
        size, error = _scan_single_cache_dir(nonexistent, repos)
        assert size == 0
        assert error is None
        assert repos == {}

    def test_deduplication_preserves_first_entry(self, tmp_path: Path) -> None:
        """Existing entries in repos dict should not be overwritten."""
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        existing_model = InstalledModel(
            repo_id="org/repo",
            size_on_disk_mb=100,
            repo_type="model",
            last_accessed=None,
            revisions=["existing"],
        )
        repos: dict = {("model", "org/repo"): existing_model}

        # Mock scan_cache_dir to return same repo with different data
        mock_repo = MagicMock()
        mock_repo.repo_id = "org/repo"
        mock_repo.repo_type = "model"
        mock_repo.size_on_disk = 200 * 1024 * 1024  # Different size
        mock_repo.revisions = []

        scan_result = MagicMock()
        scan_result.size_on_disk = mock_repo.size_on_disk
        scan_result.repos = [mock_repo]

        with patch("huggingface_hub.scan_cache_dir", return_value=scan_result):
            size, error = _scan_single_cache_dir(cache_dir, repos)

        # Original entry should be preserved
        assert repos[("model", "org/repo")].size_on_disk_mb == 100
        assert repos[("model", "org/repo")].revisions == ["existing"]
        assert error is None

    def test_revision_timestamp_parsing(self, tmp_path: Path) -> None:
        """Revision last_modified timestamps should be parsed correctly."""
        from datetime import UTC, datetime

        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        mock_revision = MagicMock()
        mock_revision.commit_hash = "abc12345678"
        mock_revision.last_modified = 1704067200.0  # 2024-01-01 00:00:00 UTC

        mock_repo = MagicMock()
        mock_repo.repo_id = "org/model"
        mock_repo.repo_type = "model"
        mock_repo.size_on_disk = 100 * 1024 * 1024
        mock_repo.revisions = [mock_revision]

        scan_result = MagicMock()
        scan_result.size_on_disk = mock_repo.size_on_disk
        scan_result.repos = [mock_repo]

        repos: dict = {}
        with patch("huggingface_hub.scan_cache_dir", return_value=scan_result):
            _scan_single_cache_dir(cache_dir, repos)

        model = repos[("model", "org/model")]
        assert model.last_accessed == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert model.revisions == ["abc12345"]

    def test_revision_with_none_timestamp(self, tmp_path: Path) -> None:
        """Revision with None last_modified should be handled gracefully."""
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir()

        mock_revision = MagicMock()
        mock_revision.commit_hash = "abc12345678"
        mock_revision.last_modified = None

        mock_repo = MagicMock()
        mock_repo.repo_id = "org/model"
        mock_repo.repo_type = "model"
        mock_repo.size_on_disk = 50 * 1024 * 1024
        mock_repo.revisions = [mock_revision]

        scan_result = MagicMock()
        scan_result.size_on_disk = mock_repo.size_on_disk
        scan_result.repos = [mock_repo]

        repos: dict = {}
        with patch("huggingface_hub.scan_cache_dir", return_value=scan_result):
            _scan_single_cache_dir(cache_dir, repos)

        model = repos[("model", "org/model")]
        assert model.last_accessed is None
        assert model.revisions == ["abc12345"]


class TestMultiCacheScanning:
    """Tests for multi-cache directory scanning in scan_hf_cache."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_cache()

    def test_scans_both_hub_and_transformers_cache(self, tmp_path: Path) -> None:
        """Should scan both HF hub cache and transformers cache when different."""
        hub_cache = tmp_path / "hub"
        hub_cache.mkdir()
        tf_cache = tmp_path / "transformers"
        tf_cache.mkdir()

        # Model in hub cache
        hub_repo = MagicMock()
        hub_repo.repo_id = "hub/model"
        hub_repo.repo_type = "model"
        hub_repo.size_on_disk = 100 * 1024 * 1024
        hub_repo.revisions = []

        hub_scan = MagicMock()
        hub_scan.size_on_disk = hub_repo.size_on_disk
        hub_scan.repos = [hub_repo]

        # Model in transformers cache
        tf_repo = MagicMock()
        tf_repo.repo_id = "tf/model"
        tf_repo.repo_type = "model"
        tf_repo.size_on_disk = 200 * 1024 * 1024
        tf_repo.revisions = []

        tf_scan = MagicMock()
        tf_scan.size_on_disk = tf_repo.size_on_disk
        tf_scan.repos = [tf_repo]

        def mock_scan(path: Path) -> MagicMock:
            if path == hub_cache:
                return hub_scan
            return tf_scan

        with (
            patch("shared.model_manager.hf_cache.resolve_hf_cache_dir", return_value=hub_cache),
            patch("shared.model_manager.hf_cache.resolve_transformers_cache_dir", return_value=tf_cache),
            patch("huggingface_hub.scan_cache_dir", side_effect=mock_scan),
        ):
            result = scan_hf_cache(force_refresh=True)

        assert ("model", "hub/model") in result.repos
        assert ("model", "tf/model") in result.repos
        assert result.total_size_mb == 300

    def test_deduplication_prefers_hub_cache(self, tmp_path: Path) -> None:
        """When same model in both caches, hub cache entry should be used."""
        hub_cache = tmp_path / "hub"
        hub_cache.mkdir()
        tf_cache = tmp_path / "transformers"
        tf_cache.mkdir()

        # Same model in hub cache (scanned first)
        hub_repo = MagicMock()
        hub_repo.repo_id = "org/model"
        hub_repo.repo_type = "model"
        hub_repo.size_on_disk = 100 * 1024 * 1024
        hub_repo.revisions = []

        hub_scan = MagicMock()
        hub_scan.size_on_disk = hub_repo.size_on_disk
        hub_scan.repos = [hub_repo]

        # Same model in transformers cache with different size
        tf_repo = MagicMock()
        tf_repo.repo_id = "org/model"
        tf_repo.repo_type = "model"
        tf_repo.size_on_disk = 200 * 1024 * 1024
        tf_repo.revisions = []

        tf_scan = MagicMock()
        tf_scan.size_on_disk = tf_repo.size_on_disk
        tf_scan.repos = [tf_repo]

        def mock_scan(path: Path) -> MagicMock:
            if path == hub_cache:
                return hub_scan
            return tf_scan

        with (
            patch("shared.model_manager.hf_cache.resolve_hf_cache_dir", return_value=hub_cache),
            patch("shared.model_manager.hf_cache.resolve_transformers_cache_dir", return_value=tf_cache),
            patch("huggingface_hub.scan_cache_dir", side_effect=mock_scan),
        ):
            result = scan_hf_cache(force_refresh=True)

        # Should have hub cache version (100 MB, not 200 MB)
        assert result.repos[("model", "org/model")].size_on_disk_mb == 100

    def test_transformers_cache_not_scanned_when_same_as_hub(self, tmp_path: Path) -> None:
        """When transformers cache equals hub cache, only scan once."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        mock_repo = MagicMock()
        mock_repo.repo_id = "org/model"
        mock_repo.repo_type = "model"
        mock_repo.size_on_disk = 100 * 1024 * 1024
        mock_repo.revisions = []

        scan_result = MagicMock()
        scan_result.size_on_disk = mock_repo.size_on_disk
        scan_result.repos = [mock_repo]

        scan_call_count = 0

        def mock_scan(_path: Path) -> MagicMock:
            nonlocal scan_call_count
            scan_call_count += 1
            return scan_result

        with (
            patch("shared.model_manager.hf_cache.resolve_hf_cache_dir", return_value=cache_dir),
            patch("shared.model_manager.hf_cache.resolve_transformers_cache_dir", return_value=None),
            patch("huggingface_hub.scan_cache_dir", side_effect=mock_scan),
        ):
            result = scan_hf_cache(force_refresh=True)

        # Should only scan once since transformers cache returns None
        assert scan_call_count == 1
        assert ("model", "org/model") in result.repos
