#!/usr/bin/env python3
"""Clean the test PostgreSQL database prior to running integration tests."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from collections.abc import Sequence
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def _confirm_reset(force: bool) -> None:
    env_force = os.getenv("CONFIRM_TEST_DB_RESET", "").lower() in {"1", "true", "yes"}
    if force or env_force:
        return

    prompt = "WARNING: this will truncate all tables in the configured test database. " "Type 'RESET' to continue: "
    try:
        response = input(prompt)
    except EOFError:  # pragma: no cover - defensive
        raise SystemExit("Input aborted. Database reset cancelled.") from None

    if response.strip().upper() != "RESET":
        raise SystemExit("Database reset cancelled by user.")


def _get_database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("TEST_DATABASE_URL or DATABASE_URL must be set to clean the test database.")

    if not url.startswith("postgres"):
        raise SystemExit("reset_test_db.py currently supports PostgreSQL URLs only.")

    if "+asyncpg" not in url:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    return url


async def _truncate_all_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename;
                """
            )
        )

        tables = [row[0] for row in result]
        if not tables:
            return

        quoted = ", ".join(f'"{name}"' for name in tables)
        await conn.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE;"))


async def _perform_reset(database_url: str) -> None:
    engine = create_async_engine(database_url, isolation_level="AUTOCOMMIT")
    try:
        await _truncate_all_tables(engine)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Truncate all tables in the test database.")
    parser.add_argument("--force", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args(argv)

    _confirm_reset(force=args.force)
    database_url = _get_database_url()

    asyncio.run(_perform_reset(database_url))
    print("✅ Test database cleaned.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
