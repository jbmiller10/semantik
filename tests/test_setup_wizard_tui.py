import os
import re
import stat
from pathlib import Path

import pytest

from docker_setup_tui import FLOWER_PASSWORD_SYMBOLS, DockerSetupTUI, generate_flower_credentials, mask_secret


def test_generate_flower_credentials_are_strong_and_parseable() -> None:
    username, password = generate_flower_credentials()

    assert re.fullmatch(r"flower_[0-9a-f]{8}", username)
    assert len(password) >= 16
    assert any(c.islower() for c in password)
    assert any(c.isupper() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(c in FLOWER_PASSWORD_SYMBOLS for c in password)
    assert ":" not in password
    assert "#" not in password


@pytest.mark.parametrize(
    ("value", "visible", "expected"),
    [
        ("", 4, "(unset)"),
        ("a", 4, "*"),
        ("abcd", 4, "****"),
        ("abcde", 4, "*bcde"),
        ("secretvalue", 4, "*******alue"),
    ],
)
def test_mask_secret(value: str, visible: int, expected: str) -> None:
    assert mask_secret(value, visible=visible) == expected


def test_save_env_file_updates_template_and_sets_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = "\n".join(
        [
            "JWT_SECRET_KEY=CHANGE_THIS_TO_A_STRONG_SECRET_KEY",
            "ACCESS_TOKEN_EXPIRE_MINUTES=1440",
            "FLOWER_USERNAME=replace-me-with-flower-user",
            "FLOWER_PASSWORD=replace-me-with-strong-flower-password",
            "CONNECTOR_SECRETS_KEY=CHANGE_THIS_TO_A_FERNET_KEY",
            "DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B",
            "DEFAULT_QUANTIZATION=float16",
            "ENABLE_LOCAL_LLM=true",
            "DEFAULT_LLM_QUANTIZATION=int8",
            "LLM_UNLOAD_AFTER_SECONDS=300",
            "LLM_KV_CACHE_BUFFER_MB=1024",
            "LLM_TRUST_REMOTE_CODE=false",
            "POSTGRES_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD",
            "REDIS_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD",
            "QDRANT_API_KEY=CHANGE_THIS_TO_A_STRONG_API_KEY",
            "DEFAULT_COLLECTION=work_docs",
            "DOCUMENT_PATH=./documents",
            "WEBUI_WORKERS=1",
            "LOG_LEVEL=INFO",
            "HF_CACHE_DIR=./models",
            "HF_HUB_OFFLINE=false",
            "",
        ]
    )

    monkeypatch.chdir(tmp_path)
    Path(".env.docker.example").write_text(template, encoding="utf-8")

    tui = DockerSetupTUI()
    tui.config = {
        "USE_GPU": "false",
        "POSTGRES_DB": "semantik",
        "POSTGRES_USER": "semantik",
        "POSTGRES_HOST": "postgres",
        "POSTGRES_PORT": "5432",
        "POSTGRES_PASSWORD": "p" * 64,
        "JWT_SECRET_KEY": "a" * 64,
        "REDIS_PASSWORD": "b" * 64,
        "QDRANT_API_KEY": "c" * 64,
        "CONNECTOR_SECRETS_KEY": "",
        "ACCESS_TOKEN_EXPIRE_MINUTES": "60",
        "DEFAULT_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "DEFAULT_QUANTIZATION": "float16",
        "ENABLE_LOCAL_LLM": "true",
        "DEFAULT_LLM_QUANTIZATION": "int8",
        "LLM_UNLOAD_AFTER_SECONDS": "300",
        "LLM_KV_CACHE_BUFFER_MB": "1024",
        "LLM_TRUST_REMOTE_CODE": "false",
        "WEBUI_WORKERS": "auto",
        "HF_CACHE_DIR": "./models",
        "HF_HUB_OFFLINE": "false",
        "LOG_LEVEL": "INFO",
        "ENVIRONMENT": "development",
        "DEFAULT_COLLECTION": "work_docs",
        "FLOWER_USERNAME": "flower_1234abcd",
        "FLOWER_PASSWORD": "Aa1!" + "x" * 20,
        "DOCUMENT_PATH": str(tmp_path / "documents"),
    }

    env_test_written = tui._save_env_file(generate_env_test=True, env_test_db_name="semantik_test")
    assert env_test_written is True

    env_text = Path(".env").read_text(encoding="utf-8")
    assert "JWT_SECRET_KEY=" + ("a" * 64) in env_text
    assert "POSTGRES_PASSWORD=" + ("p" * 64) in env_text
    assert "REDIS_PASSWORD=" + ("b" * 64) in env_text
    assert "QDRANT_API_KEY=" + ("c" * 64) in env_text
    assert "CONNECTOR_SECRETS_KEY=" in env_text
    assert "WEBUI_WORKERS=auto" in env_text
    assert "FLOWER_USERNAME=flower_1234abcd" in env_text
    assert "FLOWER_PASSWORD=Aa1!" in env_text
    assert "DOCUMENT_PATH=" + str(tmp_path / "documents") in env_text

    env_test_text = Path(".env.test").read_text(encoding="utf-8")
    assert "POSTGRES_HOST=localhost" in env_test_text
    assert "POSTGRES_DB=semantik_test" in env_test_text
    assert "DATABASE_URL=postgresql://" in env_test_text

    if os.name != "nt":
        env_mode = stat.S_IMODE(Path(".env").stat().st_mode)
        assert env_mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
        env_test_mode = stat.S_IMODE(Path(".env.test").stat().st_mode)
        assert env_test_mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
