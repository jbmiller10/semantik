"""Unit tests for plugin security module."""

from __future__ import annotations

import logging
from unittest.mock import patch

from shared.plugins.security import (
    SENSITIVE_ENV_PATTERNS,
    _sanitize_audit_details,
    audit_log,
    get_sanitized_environment,
)


class TestGetSanitizedEnvironment:
    """Tests for get_sanitized_environment function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_sanitized_environment()
        assert isinstance(result, dict)

    def test_excludes_password_vars(self, monkeypatch):
        """Should exclude variables containing PASSWORD."""
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        monkeypatch.setenv("NORMAL_VAR", "value")

        result = get_sanitized_environment()

        assert "DB_PASSWORD" not in result
        assert "NORMAL_VAR" in result

    def test_excludes_secret_vars(self, monkeypatch):
        """Should exclude variables containing SECRET."""
        monkeypatch.setenv("JWT_SECRET_KEY", "supersecret")
        monkeypatch.setenv("APP_NAME", "test")

        result = get_sanitized_environment()

        assert "JWT_SECRET_KEY" not in result
        assert "APP_NAME" in result

    def test_excludes_token_vars(self, monkeypatch):
        """Should exclude variables containing TOKEN."""
        monkeypatch.setenv("AUTH_TOKEN", "tok123")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_xxx")

        result = get_sanitized_environment()

        assert "AUTH_TOKEN" not in result
        assert "GITHUB_TOKEN" not in result

    def test_excludes_api_key_vars(self, monkeypatch):
        """Should exclude variables containing API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")

        result = get_sanitized_environment()

        assert "OPENAI_API_KEY" not in result

    def test_excludes_key_vars(self, monkeypatch):
        """Should exclude variables containing KEY."""
        monkeypatch.setenv("ENCRYPTION_KEY", "abc123")

        result = get_sanitized_environment()

        assert "ENCRYPTION_KEY" not in result

    def test_case_insensitive(self, monkeypatch):
        """Should be case insensitive when matching patterns."""
        monkeypatch.setenv("my_password", "secret")
        monkeypatch.setenv("MyPassword", "secret2")

        result = get_sanitized_environment()

        assert "my_password" not in result
        assert "MyPassword" not in result

    def test_includes_path(self, monkeypatch):
        """Should include non-sensitive vars like PATH."""
        monkeypatch.setenv("PATH", "/usr/bin")

        result = get_sanitized_environment()

        assert "PATH" in result

    def test_all_patterns_excluded(self, monkeypatch):
        """Should exclude all sensitive patterns."""
        for pattern in SENSITIVE_ENV_PATTERNS:
            var_name = f"TEST_{pattern}_VAR"
            monkeypatch.setenv(var_name, "secret")

        result = get_sanitized_environment()

        for pattern in SENSITIVE_ENV_PATTERNS:
            var_name = f"TEST_{pattern}_VAR"
            assert var_name not in result


class TestAuditLog:
    """Tests for audit_log function."""

    def test_logs_at_info_level(self, caplog):
        """Should log at INFO level by default."""
        with caplog.at_level(logging.INFO):
            audit_log("test-plugin", "plugin.registered")

        assert "PLUGIN_AUDIT: test-plugin - plugin.registered" in caplog.text

    def test_logs_at_custom_level(self, caplog):
        """Should log at specified level."""
        with caplog.at_level(logging.WARNING):
            audit_log("test-plugin", "plugin.load.failed", level=logging.WARNING)

        assert "PLUGIN_AUDIT: test-plugin - plugin.load.failed" in caplog.text

    def test_includes_plugin_id_in_extra(self, caplog):
        """Should include plugin_id in log extra."""
        with caplog.at_level(logging.INFO):
            audit_log("my-plugin", "test.action")

        # caplog.records contains the LogRecord objects
        assert any(getattr(r, "plugin_id", None) == "my-plugin" for r in caplog.records)

    def test_includes_action_in_extra(self, caplog):
        """Should include audit_action in log extra."""
        with caplog.at_level(logging.INFO):
            audit_log("test-plugin", "plugin.enabled")

        assert any(getattr(r, "audit_action", None) == "plugin.enabled" for r in caplog.records)

    def test_includes_details_in_extra(self, caplog):
        """Should include details nested under audit_details in log extra."""
        with caplog.at_level(logging.INFO):
            audit_log("test-plugin", "test.action", {"plugin_type": "embedding"})

        record = [r for r in caplog.records if "PLUGIN_AUDIT" in r.message][0]
        audit_details = getattr(record, "audit_details", None)
        assert audit_details is not None
        assert audit_details.get("plugin_type") == "embedding"

    def test_sanitizes_sensitive_details(self, caplog):
        """Should sanitize sensitive keys in details."""
        with caplog.at_level(logging.INFO):
            audit_log("test-plugin", "test.action", {"password": "secret", "name": "test"})

        record = [r for r in caplog.records if "PLUGIN_AUDIT" in r.message][0]
        audit_details = getattr(record, "audit_details", {})
        assert "password" not in audit_details
        assert audit_details.get("name") == "test"

    def test_never_raises(self, caplog):
        """Should never raise exceptions."""
        # Even with weird input, should not raise
        audit_log("test", "action", {"value": object()})
        # If we get here without exception, test passes

    def test_handles_exception_gracefully(self, caplog):
        """Should log warning on failure."""
        with patch("shared.plugins.security.logger.log", side_effect=Exception("test error")):
            with caplog.at_level(logging.WARNING):
                audit_log("test", "action")

        assert "Failed to log plugin audit" in caplog.text

    def test_includes_timestamp_in_extra(self, caplog):
        """Should include ISO timestamp in log extra."""
        with caplog.at_level(logging.INFO):
            audit_log("test-plugin", "test.action")

        record = [r for r in caplog.records if "PLUGIN_AUDIT" in r.message][0]
        timestamp = getattr(record, "audit_timestamp", None)
        assert timestamp is not None
        # Should be ISO format with timezone
        assert "T" in timestamp


class TestSanitizeAuditDetails:
    """Tests for _sanitize_audit_details function."""

    def test_returns_none_for_none(self):
        """Should return None for None input."""
        assert _sanitize_audit_details(None) is None

    def test_returns_empty_for_empty(self):
        """Should return empty dict for empty dict."""
        result = _sanitize_audit_details({})
        assert result == {}

    def test_removes_password_keys(self):
        """Should remove keys containing password."""
        result = _sanitize_audit_details({
            "db_password": "secret",
            "username": "admin",
        })

        assert "db_password" not in result
        assert result["username"] == "admin"

    def test_removes_secret_keys(self):
        """Should remove keys containing secret."""
        result = _sanitize_audit_details({
            "jwt_secret": "abc123",
            "value": "normal",
        })

        assert "jwt_secret" not in result
        assert result["value"] == "normal"

    def test_removes_token_keys(self):
        """Should remove keys containing token."""
        result = _sanitize_audit_details({
            "access_token": "xyz",
            "name": "test",
        })

        assert "access_token" not in result
        assert result["name"] == "test"

    def test_removes_api_key_keys(self):
        """Should remove keys containing api_key."""
        result = _sanitize_audit_details({
            "openai_api_key": "sk-xxx",
            "model": "gpt-4",
        })

        assert "openai_api_key" not in result
        assert result["model"] == "gpt-4"

    def test_sanitizes_nested_dicts(self):
        """Should sanitize nested dictionaries."""
        result = _sanitize_audit_details({
            "outer": {
                "api_key": "secret",
                "name": "test",
            }
        })

        assert "api_key" not in result["outer"]
        assert result["outer"]["name"] == "test"

    def test_sanitizes_lists(self):
        """Should sanitize dicts in lists."""
        result = _sanitize_audit_details({
            "items": [
                {"token": "secret", "id": 1},
                {"token": "secret2", "id": 2},
            ]
        })

        for item in result["items"]:
            assert "token" not in item
            assert "id" in item

    def test_handles_circular_reference(self):
        """Should handle circular references."""
        data: dict = {"a": 1}
        data["self"] = data  # Circular reference

        result = _sanitize_audit_details(data)

        # Should not raise, should mark circular ref
        assert result["a"] == 1
        assert result["self"] == {"__circular_reference__": True}

    def test_preserves_non_sensitive_values(self):
        """Should preserve non-sensitive values."""
        data = {
            "plugin_id": "test",
            "plugin_type": "embedding",
            "version": "1.0.0",
        }

        result = _sanitize_audit_details(data)

        assert result == data

    def test_handles_non_string_keys(self):
        """Should handle dictionaries gracefully."""
        result = _sanitize_audit_details({
            "count": 42,
            "enabled": True,
            "items": [1, 2, 3],
        })

        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["items"] == [1, 2, 3]

    def test_deeply_nested(self):
        """Should handle deeply nested structures."""
        result = _sanitize_audit_details({
            "level1": {
                "level2": {
                    "level3": {
                        "password": "secret",
                        "value": "ok",
                    }
                }
            }
        })

        assert "password" not in result["level1"]["level2"]["level3"]
        assert result["level1"]["level2"]["level3"]["value"] == "ok"
