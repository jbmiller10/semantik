#!/usr/bin/env python3
"""Utility helpers for validating environment secrets.

The validation logic is intentionally standalone so it can be reused by
entrypoints, CI checks and other scripts without pulling in the rest of the
application context. The script can be invoked directly or imported from
pytest-based unit tests.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence


# ---------------------------------------------------------------------------
# Placeholder configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaceholderRule:
    env_var: str
    placeholders: tuple[str, ...]
    message: str
    case_insensitive: bool = False

    def matches(self, value: str | None) -> bool:
        if value is None:
            return False
        candidate = value.strip()
        if not candidate:
            return False
        if self.case_insensitive:
            candidate = candidate.casefold()
            placeholders = tuple(p.casefold() for p in self.placeholders)
        else:
            placeholders = self.placeholders
        return candidate in placeholders


PLACEHOLDER_RULES: tuple[PlaceholderRule, ...] = (
    PlaceholderRule(
        env_var="JWT_SECRET_KEY",
        placeholders=("CHANGE_THIS_TO_A_STRONG_SECRET_KEY",),
        message="Generate a new secret with `openssl rand -hex 32` and update the environment.",
    ),
    PlaceholderRule(
        env_var="INTERNAL_API_KEY",
        placeholders=(
            "your-internal-api-key-here",
            "CHANGE_THIS_TO_A_STRONG_INTERNAL_API_KEY",
            "INTERNAL_API_KEY_PLACEHOLDER",
        ),
        message="Populate INTERNAL_API_KEY via wizard.sh or provide a generated key.",
    ),
    PlaceholderRule(
        env_var="POSTGRES_PASSWORD",
        placeholders=("CHANGE_THIS_TO_A_STRONG_PASSWORD",),
        message="Set a strong Postgres password (see .env.docker.example guidance).",
    ),
    PlaceholderRule(
        env_var="FLOWER_USERNAME",
        placeholders=("admin",),
        message="Flower username must not be the default `admin`.",
        case_insensitive=True,
    ),
    PlaceholderRule(
        env_var="FLOWER_PASSWORD",
        placeholders=("admin",),
        message="Flower password must not remain `admin`; set a unique password.",
        case_insensitive=True,
    ),
)

FLOWER_BASIC_AUTH_ENV = "FLOWER_BASIC_AUTH"
FLOWER_BASIC_AUTH_PLACEHOLDERS = (
    "admin:admin",
    "user:password",
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def load_env_file(path: Path) -> dict[str, str]:
    """Load key=value pairs from a .env style file.

    The parser is intentionally small – it strips comments, supports quoted
    values, and ignores blank lines. Duplicate keys later in the file override
    earlier ones.
    """

    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")

    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        env[key] = value
    return env


def merge_env_sources(env_file_values: Mapping[str, str], cli_env: Mapping[str, str]) -> dict[str, str]:
    """Merge environment sources with live environment taking precedence."""

    merged: dict[str, str] = dict(env_file_values)
    for key, value in cli_env.items():
        merged[key] = value
    return merged


def _is_weak_secret(value: str, minimum_length: int = 16) -> bool:
    if len(value) < minimum_length:
        return True
    categories = {
        "upper": any(c.isupper() for c in value),
        "lower": any(c.islower() for c in value),
        "digit": any(c.isdigit() for c in value),
        "symbol": any(not c.isalnum() for c in value),
    }
    return sum(categories.values()) < 3


def detect_placeholder_issues(env: Mapping[str, str]) -> list[str]:
    """Return a list of validation error messages for placeholder secrets."""

    errors: list[str] = []

    for rule in PLACEHOLDER_RULES:
        raw_value = env.get(rule.env_var)
        if rule.matches(raw_value):
            errors.append(f"{rule.env_var}: {rule.message}")
        elif raw_value:
            # Apply simple strength checks to secrets/passwords
            if rule.env_var in {"JWT_SECRET_KEY", "INTERNAL_API_KEY"} and _is_weak_secret(raw_value):
                errors.append(f"{rule.env_var}: value is too weak – generate a stronger secret.")
            if rule.env_var == "POSTGRES_PASSWORD" and _is_weak_secret(raw_value, minimum_length=12):
                errors.append(f"{rule.env_var}: password is too weak – use at least 12 mixed characters.")
            if rule.env_var == "FLOWER_PASSWORD" and _is_weak_secret(raw_value, minimum_length=12):
                errors.append(f"{rule.env_var}: password is too weak – use at least 12 mixed characters.")

    # Special handling for FLOWER_BASIC_AUTH
    basic_auth = env.get(FLOWER_BASIC_AUTH_ENV)
    if basic_auth:
        user, sep, password = basic_auth.partition(":")
        if not sep:
            errors.append(
                f"{FLOWER_BASIC_AUTH_ENV}: must be in username:password format; received '{basic_auth}'."
            )
        else:
            if f"{user}:{password}".lower() in FLOWER_BASIC_AUTH_PLACEHOLDERS:
                errors.append(
                    f"{FLOWER_BASIC_AUTH_ENV}: do not use default credentials ('{basic_auth}')."
                )
            env_user = env.get("FLOWER_USERNAME")
            env_password = env.get("FLOWER_PASSWORD")
            if env_user and env_user != user:
                errors.append(
                    f"FLOWER_USERNAME '{env_user}' does not match username from {FLOWER_BASIC_AUTH_ENV}."
                )
            if env_password and env_password != password:
                errors.append(
                    f"FLOWER_PASSWORD does not match password from {FLOWER_BASIC_AUTH_ENV}."
                )

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate environment secrets.")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional path to a .env file to validate (variables may be overridden by current environment).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when validation errors are detected (default: exit 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all variables that were checked and their status.",
    )
    return parser


def format_errors(errors: Iterable[str]) -> str:
    return "\n".join(f" - {msg}" for msg in errors)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    file_env: dict[str, str] = {}
    if args.env_file is not None:
        try:
            file_env = load_env_file(args.env_file)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2

    merged_env = merge_env_sources(file_env, os.environ)
    errors = detect_placeholder_issues(merged_env)

    if not errors:
        if args.verbose:
            print("✅ Environment secrets look good.")
        return 0

    print("❌ Placeholder or weak secrets detected:")
    print(format_errors(errors))

    if args.strict:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

