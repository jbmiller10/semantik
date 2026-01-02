"""Tests for plugin version compatibility checking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from shared.plugins.compatibility import (
    check_compatibility,
    get_semantik_version,
    is_compatible,
)


class TestGetSemantikVersion:
    """Tests for get_semantik_version function."""

    def test_returns_string(self) -> None:
        """Should return a version string."""
        version = get_semantik_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_fallback_version_format(self) -> None:
        """Fallback version should be valid semver."""
        # When package metadata isn't available, should return fallback
        with patch("importlib.metadata.version", side_effect=Exception):
            version = get_semantik_version()
            assert version == "2.0.0"


class TestCheckCompatibility:
    """Tests for check_compatibility function."""

    def test_no_constraint_is_compatible(self) -> None:
        """Plugin with no version constraint should always be compatible."""
        is_compat, msg = check_compatibility(None)
        assert is_compat is True
        assert msg is None

    def test_empty_constraint_is_compatible(self) -> None:
        """Plugin with empty string constraint should always be compatible."""
        is_compat, msg = check_compatibility("")
        assert is_compat is True
        assert msg is None

    def test_compatible_version_equal(self) -> None:
        """Should be compatible when versions are equal."""
        is_compat, msg = check_compatibility("2.0.0", "2.0.0")
        assert is_compat is True
        assert msg is None

    def test_compatible_version_higher(self) -> None:
        """Should be compatible when running version is higher."""
        is_compat, msg = check_compatibility("2.0.0", "2.1.0")
        assert is_compat is True
        assert msg is None

    def test_compatible_version_much_higher(self) -> None:
        """Should be compatible when running version is much higher."""
        is_compat, msg = check_compatibility("1.0.0", "3.0.0")
        assert is_compat is True
        assert msg is None

    def test_incompatible_version_lower(self) -> None:
        """Should not be compatible when running version is lower."""
        is_compat, msg = check_compatibility("2.0.0", "1.9.0")
        assert is_compat is False
        assert msg is not None
        assert "Requires Semantik >= 2.0.0" in msg
        assert "running 1.9.0" in msg

    def test_incompatible_version_slightly_lower(self) -> None:
        """Should not be compatible when patch version is lower."""
        is_compat, msg = check_compatibility("2.0.1", "2.0.0")
        assert is_compat is False
        assert msg is not None

    def test_uses_default_version_when_none(self) -> None:
        """Should use current Semantik version when not specified."""
        is_compat, msg = check_compatibility("0.0.1")
        # Should be compatible since any current version > 0.0.1
        assert is_compat is True
        assert msg is None

    def test_invalid_version_format(self) -> None:
        """Should handle invalid version format gracefully."""
        is_compat, msg = check_compatibility("2.0.0", "not-a-version")
        assert is_compat is False
        assert msg is not None
        assert "Invalid version format" in msg

    def test_invalid_constraint_format(self) -> None:
        """Should handle invalid constraint format gracefully."""
        is_compat, msg = check_compatibility("not-valid", "2.0.0")
        assert is_compat is False
        assert msg is not None
        # The packaging library wraps it with ">=" so error is about specifier
        assert "Invalid" in msg

    def test_prerelease_version(self) -> None:
        """Should handle prerelease versions."""
        # Prerelease versions are lower than release versions
        is_compat, msg = check_compatibility("2.0.0", "2.0.0rc1")
        assert is_compat is False
        assert msg is not None

    def test_prerelease_constraint(self) -> None:
        """Should handle prerelease in constraint."""
        is_compat, msg = check_compatibility("2.0.0rc1", "2.0.0")
        assert is_compat is True
        assert msg is None


class TestIsCompatible:
    """Tests for is_compatible convenience function."""

    def test_returns_true_for_compatible(self) -> None:
        """Should return True when compatible."""
        result = is_compatible("2.0.0", "2.1.0")
        assert result is True

    def test_returns_false_for_incompatible(self) -> None:
        """Should return False when incompatible."""
        result = is_compatible("3.0.0", "2.0.0")
        assert result is False

    def test_returns_true_for_no_constraint(self) -> None:
        """Should return True when no constraint."""
        result = is_compatible(None)
        assert result is True

    def test_returns_false_for_invalid_version(self) -> None:
        """Should return False for invalid versions."""
        result = is_compatible("2.0.0", "invalid")
        assert result is False


class TestVersionEdgeCases:
    """Edge case tests for version compatibility."""

    @pytest.mark.parametrize(
        ("constraint", "running", "expected"),
        [
            # Basic semver
            ("1.0.0", "1.0.0", True),
            ("1.0.0", "2.0.0", True),
            ("2.0.0", "1.0.0", False),
            # Patch versions
            ("1.0.0", "1.0.1", True),
            ("1.0.1", "1.0.0", False),
            # Minor versions
            ("1.0.0", "1.1.0", True),
            ("1.1.0", "1.0.0", False),
            # Major versions
            ("1.0.0", "2.0.0", True),
            ("2.0.0", "1.0.0", False),
        ],
    )
    def test_version_comparisons(
        self, constraint: str, running: str, expected: bool
    ) -> None:
        """Test various version comparisons."""
        result = is_compatible(constraint, running)
        assert result is expected

    def test_development_version(self) -> None:
        """Should handle development versions."""
        # Development versions like "2.0.0.dev1" should work
        is_compat, _ = check_compatibility("2.0.0", "2.0.0.dev1")
        # Dev versions are lower than release
        assert is_compat is False

    def test_local_version(self) -> None:
        """Should handle local versions."""
        # Local versions like "2.0.0+local" should work
        is_compat, _ = check_compatibility("2.0.0", "2.0.0+local")
        assert is_compat is True
