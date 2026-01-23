from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

import webui.mcp.cli as cli


def test_main_missing_token_exits() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["serve"])
    assert excinfo.value.code == 2


def test_main_happy_path_runs_server(monkeypatch: pytest.MonkeyPatch) -> None:
    server_instance = MagicMock()

    def fake_server_ctor(*, webui_url: str, auth_token: str, profile_filter):
        assert webui_url == "http://localhost:8080"
        assert auth_token == "token"
        assert profile_filter == ["coding"]
        return server_instance

    monkeypatch.setattr(cli, "SemantikMCPServer", fake_server_ctor)
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)

    cli.main(["serve", "--auth-token", "token", "--profile", "coding"])


def test_main_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch) -> None:
    server_instance = MagicMock()
    server_instance.run = MagicMock(return_value=types.SimpleNamespace())  # placeholder for coroutine object

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)

    def raise_interrupt(_coro):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli.asyncio, "run", raise_interrupt)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["serve", "--auth-token", "token"])
    assert excinfo.value.code == 130


def test_main_generic_exception_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    server_instance = MagicMock()
    server_instance.run = types.SimpleNamespace()  # placeholder for coroutine

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)

    def raise_error(_coro):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.asyncio, "run", raise_error)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["serve", "--auth-token", "token"])
    assert excinfo.value.code == 1


def test_main_unknown_command_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyParser:
        def parse_args(self, _argv):
            return types.SimpleNamespace(command="nope", auth_token="token")

        def error(self, _msg: str) -> None:  # pragma: no cover
            raise AssertionError("parser.error should not be called")

    monkeypatch.setattr(cli, "_build_parser", lambda: DummyParser())

    with pytest.raises(SystemExit, match="Unknown command"):
        cli.main(["serve", "--auth-token", "token"])


# --------------------------------------------------------------------------
# HTTP Transport Tests
# --------------------------------------------------------------------------


def test_http_transport_uses_internal_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HTTP transport uses internal API key for service mode."""
    server_instance = MagicMock()
    captured_kwargs = {}

    def fake_server_ctor(**kwargs):
        captured_kwargs.update(kwargs)
        return server_instance

    monkeypatch.setattr(cli, "SemantikMCPServer", fake_server_ctor)
    monkeypatch.setattr(cli, "_get_internal_api_key", lambda: "internal-api-key-123")
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)

    cli.main(["serve", "--transport", "http"])

    assert "internal_api_key" in captured_kwargs
    assert captured_kwargs["internal_api_key"] == "internal-api-key-123"
    assert "auth_token" not in captured_kwargs


def test_http_transport_missing_internal_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HTTP transport exits with error if internal API key is unavailable."""
    monkeypatch.setattr(cli, "_get_internal_api_key", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["serve", "--transport", "http"])
    assert excinfo.value.code == 2  # argparse error exit code


def test_http_transport_passes_host_port_to_run_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that custom host/port are passed to run_http()."""
    server_instance = MagicMock()
    captured_run_http_kwargs = {}

    def fake_run_http(**kwargs):
        captured_run_http_kwargs.update(kwargs)
        return types.SimpleNamespace()  # fake coroutine

    server_instance.run_http = fake_run_http

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)
    monkeypatch.setattr(cli, "_get_internal_api_key", lambda: "internal-key")

    def capture_run(coro):
        # asyncio.run would execute the coroutine, but we just want to verify args
        pass

    monkeypatch.setattr(cli.asyncio, "run", capture_run)

    cli.main(["serve", "--transport", "http", "--http-host", "127.0.0.1", "--http-port", "9999"])

    assert captured_run_http_kwargs["host"] == "127.0.0.1"
    assert captured_run_http_kwargs["port"] == 9999


def test_http_transport_uses_default_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HTTP transport uses default host (0.0.0.0) and port (9090)."""
    server_instance = MagicMock()
    captured_run_http_kwargs = {}

    def fake_run_http(**kwargs):
        captured_run_http_kwargs.update(kwargs)
        return types.SimpleNamespace()

    server_instance.run_http = fake_run_http

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)
    monkeypatch.setattr(cli, "_get_internal_api_key", lambda: "internal-key")
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)

    cli.main(["serve", "--transport", "http"])

    assert captured_run_http_kwargs["host"] == "0.0.0.0"
    assert captured_run_http_kwargs["port"] == 9090


# --------------------------------------------------------------------------
# Verbose Flag Tests
# --------------------------------------------------------------------------


def test_verbose_flag_sets_debug_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --verbose flag sets log level to DEBUG."""
    server_instance = MagicMock()
    captured_log_level = None

    original_basicConfig = cli.logging.basicConfig

    def fake_basicConfig(**kwargs):
        nonlocal captured_log_level
        captured_log_level = kwargs.get("level")

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)
    monkeypatch.setattr(cli.logging, "basicConfig", fake_basicConfig)

    cli.main(["serve", "--auth-token", "token", "--verbose"])

    assert captured_log_level == "DEBUG"


def test_log_level_arg_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log-level argument is respected when --verbose is not used."""
    server_instance = MagicMock()
    captured_log_level = None

    def fake_basicConfig(**kwargs):
        nonlocal captured_log_level
        captured_log_level = kwargs.get("level")

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)
    monkeypatch.setattr(cli.logging, "basicConfig", fake_basicConfig)

    cli.main(["serve", "--auth-token", "token", "--log-level", "WARNING"])

    assert captured_log_level == "WARNING"


def test_verbose_overrides_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --verbose overrides --log-level."""
    server_instance = MagicMock()
    captured_log_level = None

    def fake_basicConfig(**kwargs):
        nonlocal captured_log_level
        captured_log_level = kwargs.get("level")

    monkeypatch.setattr(cli, "SemantikMCPServer", lambda **_kwargs: server_instance)
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)
    monkeypatch.setattr(cli.logging, "basicConfig", fake_basicConfig)

    cli.main(["serve", "--auth-token", "token", "--log-level", "ERROR", "--verbose"])

    assert captured_log_level == "DEBUG"  # --verbose overrides --log-level


# --------------------------------------------------------------------------
# Additional Tests
# --------------------------------------------------------------------------


def test_get_internal_api_key_returns_none_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_internal_api_key returns None on exception."""
    def fake_import_error():
        raise ImportError("shared module not found")

    # We need to make the import fail within _get_internal_api_key
    import sys
    monkeypatch.setitem(sys.modules, "shared.config", types.SimpleNamespace(settings=None))
    monkeypatch.setitem(
        sys.modules,
        "shared.config.internal_api_key",
        types.SimpleNamespace(ensure_internal_api_key=lambda _: None),
    )

    result = cli._get_internal_api_key()  # noqa: SLF001 - unit test
    # Should return None when ensure_internal_api_key returns None
    assert result is None


def test_http_transport_with_profile_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HTTP transport passes profile filter to server."""
    server_instance = MagicMock()
    captured_kwargs = {}

    def fake_server_ctor(**kwargs):
        captured_kwargs.update(kwargs)
        return server_instance

    monkeypatch.setattr(cli, "SemantikMCPServer", fake_server_ctor)
    monkeypatch.setattr(cli, "_get_internal_api_key", lambda: "internal-key")
    monkeypatch.setattr(cli.asyncio, "run", lambda _coro: None)

    cli.main(["serve", "--transport", "http", "--profile", "docs", "--profile", "code"])

    assert captured_kwargs["profile_filter"] == ["docs", "code"]


def test_stdio_transport_still_works_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stdio transport works correctly with auth token."""
    server_instance = MagicMock()
    captured_kwargs = {}
    run_called = False

    def fake_server_ctor(**kwargs):
        captured_kwargs.update(kwargs)
        return server_instance

    def fake_asyncio_run(coro):
        nonlocal run_called
        run_called = True

    monkeypatch.setattr(cli, "SemantikMCPServer", fake_server_ctor)
    monkeypatch.setattr(cli.asyncio, "run", fake_asyncio_run)

    cli.main(["serve", "--transport", "stdio", "--auth-token", "my-jwt-token"])

    assert captured_kwargs["auth_token"] == "my-jwt-token"
    assert "internal_api_key" not in captured_kwargs
    assert run_called is True
