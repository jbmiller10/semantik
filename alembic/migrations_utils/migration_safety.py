"""Migration safety utilities for preventing data loss during destructive operations.

This module provides centralized safety functions that should be used in all migrations
that perform destructive operations like DROP TABLE, TRUNCATE, or DELETE.
"""

import datetime
import logging
import os
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

BACKUP_RETENTION_DAYS = int(os.getenv("MIGRATION_BACKUP_RETENTION_DAYS", "7"))


def require_destructive_flag(operation_description: str = "destructive migration") -> None:
    """Check if destructive migrations are explicitly allowed.

    Args:
        operation_description: Description of the destructive operation

    Raises:
        RuntimeError: If ALLOW_DESTRUCTIVE_MIGRATIONS is not set to 'true'
    """
    if os.getenv("ALLOW_DESTRUCTIVE_MIGRATIONS", "").lower() != "true":
        raise RuntimeError(
            f"BLOCKED: {operation_description} requires explicit permission.\n"
            f"Set environment variable ALLOW_DESTRUCTIVE_MIGRATIONS=true to proceed.\n"
            f"WARNING: This operation may result in data loss!"
        )
    logger.warning(f"DESTRUCTIVE OPERATION ALLOWED: {operation_description}")


def check_table_exists(conn: Any, table_name: str) -> bool:
    """Check if a table exists in the database.

    Args:
        conn: Database connection
        table_name: Name of the table to check

    Returns:
        True if table exists, False otherwise
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    try:
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                )
                """
            ).bindparams(table_name=table_name)
        )
        return bool(result.scalar())
    except SQLAlchemyError:
        return False


def get_table_row_count(conn: Any, table_name: str) -> int:
    """Get the number of rows in a table.

    Args:
        conn: Database connection
        table_name: Name of the table

    Returns:
        Number of rows in the table
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    try:
        conn.execute(
            text(
                """
                DO $$
                DECLARE
                    table_name TEXT := :table_name;
                    row_count BIGINT;
                BEGIN
                    EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO row_count;
                    RAISE NOTICE 'COUNT:%', row_count;
                END $$;
                """
            ).bindparams(table_name=table_name)
        )

        for notice in conn.connection.notices:
            if notice.startswith("NOTICE:  COUNT:"):
                return int(notice.split(":")[2])

        return 0
    except SQLAlchemyError as e:
        logger.error(f"Error counting rows in {table_name}: {e}")
        return 0


def create_table_backup(
    conn: Any,
    table_name: str,
    migration_revision: str,
    check_exists: bool = True,
    require_data: bool = False
) -> tuple[str | None, int]:
    """Create a timestamped backup of a table before destructive operations.

    Args:
        conn: Database connection
        table_name: Name of the table to backup
        migration_revision: Alembic revision ID for tracking
        check_exists: Whether to check if table exists first
        require_data: Whether to require the table to have data

    Returns:
        Tuple of (backup_table_name, row_count) or (None, 0) if no backup created
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    if check_exists and not check_table_exists(conn, table_name):
        logger.info(f"Table {table_name} does not exist, no backup needed")
        return None, 0

    row_count = get_table_row_count(conn, table_name)

    if require_data and row_count == 0:
        logger.info(f"Table {table_name} has no data, no backup needed")
        return None, 0

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    backup_table_name = f"{table_name}_backup_{timestamp}"

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*_backup_\d{8}_\d{6}$", backup_table_name):
        raise ValueError(f"Invalid backup table name format: {backup_table_name}")

    logger.info(f"Creating backup of {table_name} with {row_count} rows to {backup_table_name}")

    try:
        conn.execute(
            text(
                """
                DO $$
                DECLARE
                    source_table TEXT := :source_table;
                    backup_table TEXT := :backup_table;
                BEGIN
                    EXECUTE format('CREATE TABLE %I AS TABLE %I WITH DATA', backup_table, source_table);
                    RAISE NOTICE 'Backup created successfully';
                END $$;
                """
            ).bindparams(source_table=table_name, backup_table=backup_table_name)
        )

        ensure_migration_backups_table(conn)

        conn.execute(
            text(
                """
                INSERT INTO migration_backups
                (backup_table_name, original_table_name, record_count, migration_revision, retention_until)
                VALUES (:backup_table, :original_table, :count, :revision, NOW() + INTERVAL :days)
                """
            ).bindparams(
                backup_table=backup_table_name,
                original_table=table_name,
                count=row_count,
                revision=migration_revision,
                days=f"{BACKUP_RETENTION_DAYS} days"
            )
        )

        logger.info(f"Backup created successfully: {backup_table_name}")
        return backup_table_name, row_count

    except SQLAlchemyError as e:
        logger.error(f"Failed to create backup: {e}")
        raise RuntimeError(f"Backup creation failed for {table_name}: {e}") from e


def verify_backup(conn: Any, backup_table_name: str, expected_count: int) -> bool:
    """Verify that a backup table exists and has the expected number of rows.

    Args:
        conn: Database connection
        backup_table_name: Name of the backup table
        expected_count: Expected number of rows

    Returns:
        True if backup is valid, False otherwise
    """
    if not backup_table_name:
        return False

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*_backup_\d{8}_\d{6}$", backup_table_name):
        logger.error(f"Invalid backup table name format: {backup_table_name}")
        return False

    if not check_table_exists(conn, backup_table_name):
        logger.error(f"Backup table {backup_table_name} does not exist")
        return False

    actual_count = get_table_row_count(conn, backup_table_name)

    if actual_count != expected_count:
        logger.error(
            f"Backup verification failed: expected {expected_count} rows, "
            f"found {actual_count} in {backup_table_name}"
        )
        return False

    logger.info(f"Backup verified: {backup_table_name} has {actual_count} rows")
    return True


def ensure_migration_backups_table(conn: Any) -> None:
    """Ensure the migration_backups tracking table exists.

    Args:
        conn: Database connection
    """
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS migration_backups (
                id SERIAL PRIMARY KEY,
                backup_table_name VARCHAR NOT NULL,
                original_table_name VARCHAR NOT NULL,
                record_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                migration_revision VARCHAR,
                retention_until TIMESTAMPTZ
            )
            """
        )
    )


def safe_drop_table(
    conn: Any,
    table_name: str,
    migration_revision: str,
    cascade: bool = False,
    backup: bool = True
) -> tuple[str | None, int]:
    """Safely drop a table with optional backup and safety checks.

    Args:
        conn: Database connection
        table_name: Name of the table to drop
        migration_revision: Alembic revision ID
        cascade: Whether to use CASCADE
        backup: Whether to create a backup first

    Returns:
        Tuple of (backup_table_name, row_count) or (None, 0)
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    require_destructive_flag(f"DROP TABLE {table_name}")

    backup_table_name = None
    row_count = 0

    if backup and check_table_exists(conn, table_name):
        backup_table_name, row_count = create_table_backup(
            conn, table_name, migration_revision, check_exists=False
        )

        if row_count > 0 and backup_table_name and not verify_backup(conn, backup_table_name, row_count):
            raise RuntimeError(f"Backup verification failed for {table_name}, aborting DROP")

    cascade_clause = "CASCADE" if cascade else ""

    conn.execute(
        text(
            f"""
            DO $$
            DECLARE
                table_name TEXT := :table_name;
            BEGIN
                EXECUTE format('DROP TABLE IF EXISTS %I {cascade_clause}', table_name);
            END $$;
            """
        ).bindparams(table_name=table_name)
    )

    logger.info(f"Table {table_name} dropped successfully")

    if backup_table_name and row_count > 0:
        logger.info(f"Data preserved in backup table: {backup_table_name} ({row_count} rows)")

    return backup_table_name, row_count


def restore_from_backup(
    conn: Any,
    original_table_name: str,
    backup_table_name: str,
    drop_backup: bool = False
) -> bool:
    """Restore a table from its backup.

    Args:
        conn: Database connection
        original_table_name: Name of the original table
        backup_table_name: Name of the backup table
        drop_backup: Whether to drop the backup after restoration

    Returns:
        True if restoration successful, False otherwise
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", original_table_name):
        raise ValueError(f"Invalid table name: {original_table_name}")

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*_backup_\d{8}_\d{6}$", backup_table_name):
        raise ValueError(f"Invalid backup table name format: {backup_table_name}")

    if not check_table_exists(conn, backup_table_name):
        logger.error(f"Backup table {backup_table_name} does not exist")
        return False

    try:
        if check_table_exists(conn, original_table_name):
            conn.execute(
                text(
                    """
                    DO $$
                    DECLARE
                        table_name TEXT := :table_name;
                    BEGIN
                        EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', table_name);
                    END $$;
                    """
                ).bindparams(table_name=original_table_name)
            )

        conn.execute(
            text(
                """
                DO $$
                DECLARE
                    original_table TEXT := :original_table;
                    backup_table TEXT := :backup_table;
                BEGIN
                    EXECUTE format('CREATE TABLE %I AS TABLE %I WITH DATA', original_table, backup_table);
                END $$;
                """
            ).bindparams(original_table=original_table_name, backup_table=backup_table_name)
        )

        if drop_backup:
            conn.execute(
                text(
                    """
                    DO $$
                    DECLARE
                        backup_table TEXT := :backup_table;
                    BEGIN
                        EXECUTE format('DROP TABLE IF EXISTS %I', backup_table);
                    END $$;
                    """
                ).bindparams(backup_table=backup_table_name)
            )

            conn.execute(
                text(
                    """
                    DELETE FROM migration_backups
                    WHERE backup_table_name = :backup_table
                    """
                ).bindparams(backup_table=backup_table_name)
            )

        logger.info(f"Successfully restored {original_table_name} from {backup_table_name}")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Failed to restore from backup: {e}")
        return False

