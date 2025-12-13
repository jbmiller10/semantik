"""Generate a Fernet encryption key for connector secrets.

This key is used to encrypt sensitive connector credentials (passwords, tokens,
SSH keys) stored in the connector_secrets table.

Usage (print only):
    uv run python scripts/generate_secrets_key.py

Usage (write into .env or a custom file):
    uv run python scripts/generate_secrets_key.py --write --env-file .env

The generated key is a 44-character base64-encoded string suitable for Fernet
symmetric encryption (AES-128-CBC + HMAC-SHA256).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Fernet encryption key for connector secrets")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help=".env file to update")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite CONNECTOR_SECRETS_KEY in the specified env file (creates the file if missing)",
    )
    return parser.parse_args()


def _write_env(env_path: Path, key: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    key_line = f"CONNECTOR_SECRETS_KEY={key}"
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith("CONNECTOR_SECRETS_KEY="):
            lines[idx] = key_line
            replaced = True
            break

    if not replaced:
        lines.append(key_line)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_fernet_key() -> str:
    """Generate a new Fernet key.

    Returns:
        A Fernet key string (44 chars, base64-encoded 32 bytes)
    """
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode("utf-8")


def main() -> None:
    args = _parse_args()
    key = generate_fernet_key()

    if args.write:
        _write_env(args.env_file, key)
        print(f"Wrote CONNECTOR_SECRETS_KEY to {args.env_file}")
    else:
        print(key)


if __name__ == "__main__":
    main()
