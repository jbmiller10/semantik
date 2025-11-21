"""Generate a secure JWT secret for Semantik services.

Usage (print only):
    uv run python scripts/generate_jwt_secret.py

Usage (write into .env or a custom file):
    uv run python scripts/generate_jwt_secret.py --write --env-file .env
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a JWT secret key")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help=".env file to update")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite JWT_SECRET_KEY in the specified env file (creates the file if missing)",
    )
    return parser.parse_args()


def _write_env(env_path: Path, secret: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    key_line = f"JWT_SECRET_KEY={secret}"
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith("JWT_SECRET_KEY="):
            lines[idx] = key_line
            replaced = True
            break

    if not replaced:
        lines.append(key_line)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    secret = secrets.token_hex(32)

    if args.write:
        _write_env(args.env_file, secret)
        print(f"✅ Wrote JWT_SECRET_KEY to {args.env_file}")
    else:
        print(secret)


if __name__ == "__main__":
    main()
