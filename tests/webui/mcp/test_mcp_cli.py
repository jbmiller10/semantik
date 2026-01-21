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
