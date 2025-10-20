from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts import validate_env

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Sequence


@pytest.fixture()
def secure_env(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, str] | None], None]:
    """Provide a helper that seeds strong defaults for secrets before validation tests."""

    def apply(overrides: dict[str, str] | None = None) -> None:
        base_values: dict[str, str] = {
            "JWT_SECRET_KEY": "Sup3rStrongJWTSecretKey!123",
            "POSTGRES_PASSWORD": "Sup3rStrongDBPass!123",
            "INTERNAL_API_KEY": "internal-api-key-1234-STRONG!",
        }
        if overrides:
            base_values.update(overrides)

        for key, value in base_values.items():
            monkeypatch.setenv(key, value)

        # Ensure Flower-related env vars start clean unless explicitly overridden
        if not overrides or "FLOWER_USERNAME" not in overrides:
            monkeypatch.delenv("FLOWER_USERNAME", raising=False)
        if not overrides or "FLOWER_PASSWORD" not in overrides:
            monkeypatch.delenv("FLOWER_PASSWORD", raising=False)

    apply(None)
    return apply


def run_main(args: Sequence[str]) -> int:
    return validate_env.main(list(args))


def test_load_env_file_basic(tmp_path: pathlib.Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # comment
        JWT_SECRET_KEY = value
        FLOWER_USERNAME='flower'
        UNUSED=foo
        """,
        encoding="utf-8",
    )

    env = validate_env.load_env_file(env_file)
    assert env["JWT_SECRET_KEY"] == "value"
    assert env["FLOWER_USERNAME"] == "flower"
    assert env["UNUSED"] == "foo"


def test_detect_placeholder_errors_for_known_values() -> None:
    env = {
        "JWT_SECRET_KEY": "CHANGE_THIS_TO_A_STRONG_SECRET_KEY",
        "POSTGRES_PASSWORD": "CHANGE_THIS_TO_A_STRONG_PASSWORD",
        "INTERNAL_API_KEY": "your-internal-api-key-here",
        "FLOWER_USERNAME": "replace-me-with-flower-user",
        "FLOWER_PASSWORD": "replace-me-with-strong-flower-password",
    }

    errors = validate_env.detect_placeholder_issues(env)
    assert len(errors) >= 5
    summary = " ".join(errors)
    assert "JWT_SECRET_KEY" in summary
    assert "POSTGRES_PASSWORD" in summary
    assert "FLOWER_USERNAME" in summary
    assert "FLOWER_PASSWORD" in summary


def test_detect_placeholder_allows_secure_values() -> None:
    env = {
        "JWT_SECRET_KEY": "A1b!2c#3d$4e%5f^",
        "POSTGRES_PASSWORD": "Str0ngPassword!",
        "INTERNAL_API_KEY": "internal-key-123456789",
        "FLOWER_USERNAME": "floweruser",
        "FLOWER_PASSWORD": "Sup3rSecret!",
    }

    errors = validate_env.detect_placeholder_issues(env)
    assert errors == []


def test_flow_credentials_missing_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOWER_USERNAME", raising=False)
    monkeypatch.delenv("FLOWER_PASSWORD", raising=False)

    code = run_main(["--strict"])
    assert code == 1


def test_main_with_missing_file_returns_error(tmp_path: pathlib.Path) -> None:
    assert run_main(["--env-file", str(tmp_path / "missing.env")]) == 2


def test_main_reports_errors_for_placeholder_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "env"
    env_file.write_text("JWT_SECRET_KEY=CHANGE_THIS_TO_A_STRONG_SECRET_KEY\n", encoding="utf-8")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    code = run_main(["--env-file", str(env_file), "--strict"])
    assert code == 1


def test_main_returns_zero_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "A1b!2c#3d$4e%5f^")
    monkeypatch.setenv("POSTGRES_PASSWORD", "Sup3rSecret12!")
    monkeypatch.delenv("FLOWER_USERNAME", raising=False)
    monkeypatch.delenv("FLOWER_PASSWORD", raising=False)
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)

    assert run_main([]) == 0


def test_merge_env_sources_env_overrides_file() -> None:
    file_env = {"JWT_SECRET_KEY": "file"}
    merged = validate_env.merge_env_sources(file_env, {"JWT_SECRET_KEY": "env"})
    assert merged["JWT_SECRET_KEY"] == "env"


def test_is_weak_secret_helpers() -> None:
    assert validate_env._is_weak_secret("short", minimum_length=6)
    assert validate_env._is_weak_secret("aaaaaaaaaaaaaa")
    assert not validate_env._is_weak_secret("Abcdef1234!@#$xx")
    # 32-byte hex tokens (documented guidance) should be accepted
    assert not validate_env._is_weak_secret("0123456789abcdef" * 4)


def test_flower_credentials_missing_fails(secure_env: Callable[[dict[str, str] | None], None]) -> None:
    secure_env()

    code = run_main(["--strict"])
    assert code == 1


def test_flower_admin_credentials_rejected(secure_env: Callable[[dict[str, str] | None], None]) -> None:
    secure_env(
        overrides={
            "FLOWER_USERNAME": "admin",
            "FLOWER_PASSWORD": "admin",
        }
    )

    code = run_main(["--strict"])
    assert code == 1


def test_flower_strong_credentials_pass(secure_env: Callable[[dict[str, str] | None], None]) -> None:
    secure_env(
        overrides={
            "FLOWER_USERNAME": "flower_ops",
            "FLOWER_PASSWORD": "S3cureFlowerPwd!",
        }
    )

    code = run_main(["--strict"])
    assert code == 0


def test_flower_basic_auth_only_is_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOWER_USERNAME", raising=False)
    monkeypatch.delenv("FLOWER_PASSWORD", raising=False)
    monkeypatch.setenv("FLOWER_BASIC_AUTH", "inspector:S3cureFlowerPwd!")

    code = run_main(["--strict"])
    assert code == 1
