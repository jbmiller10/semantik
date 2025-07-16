#!/usr/bin/env python3
"""
SQLite database implementation for shared package.
Centralizes database operations for users and tokens.

This module contains the actual database implementation for user authentication.
Job and file-related operations have been removed as part of the collections refactor.
"""

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from passlib.context import CryptContext
from shared.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = str(settings.webui_db)
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# Password hashing context for auth functions
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Database initialization and management
def init_db() -> None:
    """Initialize SQLite database using Alembic migrations.

    IMPORTANT: This function must be called explicitly by all entry points:
    - docker-entrypoint.sh for Docker deployments
    - Application startup code for local development
    - Test fixtures for unit tests

    The automatic init_db() call on module import was removed to prevent
    circular dependency issues with Alembic.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Find the project root (where alembic.ini is located)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent

    # Verify alembic.ini exists
    alembic_ini = project_root / "alembic.ini"
    if not alembic_ini.exists():
        logger.error(f"alembic.ini not found at {alembic_ini}")
        msg = f"Cannot find alembic.ini at {alembic_ini}"
        raise FileNotFoundError(msg)

    # Check if we need to run migrations
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if alembic_version table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'")
    alembic_exists = c.fetchone() is not None

    if not alembic_exists:
        logger.info("Database not initialized, running Alembic migrations...")
        conn.close()

        # Run alembic upgrade
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            env={**os.environ, "ALEMBIC_DATABASE_URL": f"sqlite:///{DB_PATH}"},
        )

        if result.returncode != 0:
            logger.error(f"Alembic migration failed: {result.stderr}")
            raise RuntimeError(f"Database migration failed: {result.stderr}")

        logger.info("Database migrations completed successfully")
    else:
        # Database exists, no need to run migrations
        conn.close()


def init_auth_tables(_conn: sqlite3.Connection, _c: sqlite3.Cursor) -> None:
    """Legacy function kept for compatibility.

    Auth tables are now created by Alembic migrations.
    """


def reset_database() -> None:
    """Reset the database to a fresh state.

    This removes all data and recreates the schema using Alembic.
    Use with caution - this is destructive!

    This is primarily used for testing and development.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    logger.warning("Resetting database - all data will be lost!")

    # Close any existing connections
    conn = sqlite3.connect(DB_PATH)
    conn.close()

    # Delete the database file
    db_file = Path(DB_PATH)
    if db_file.exists():
        db_file.unlink()
        logger.info(f"Deleted database file: {DB_PATH}")

    # Find the project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent

    # Run alembic upgrade to recreate schema
    logger.info("Running Alembic migrations to recreate schema...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env={**os.environ, "ALEMBIC_DATABASE_URL": f"sqlite:///{DB_PATH}"},
    )

    if result.returncode != 0:
        logger.error(f"Alembic migration failed: {result.stderr}")
        raise RuntimeError(f"Database migration failed: {result.stderr}")

    logger.info("Database reset completed successfully")


def get_database_stats() -> dict[str, Any]:
    """Get database statistics for monitoring.

    Returns statistics about users and other relevant metrics.
    Collections statistics should be obtained through the new repository pattern.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    stats = {}

    # User stats
    c.execute("SELECT COUNT(*) FROM users")
    stats["total_users"] = c.fetchone()[0]

    conn.close()
    return stats


# User management functions
def get_user(username: str) -> dict[str, Any] | None:
    """Get user by username."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Get user by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def create_user(username: str, email: str, hashed_password: str, full_name: str | None = None) -> dict[str, Any]:
    """Create a new user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if user already exists
    c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if c.fetchone():
        conn.close()
        raise ValueError("User with this username or email already exists")

    # Insert new user
    now = datetime.now(UTC).isoformat()
    c.execute(
        """INSERT INTO users (username, email, full_name, hashed_password, is_active, is_superuser, created_at, updated_at)
           VALUES (?, ?, ?, ?, 1, 0, ?, ?)""",
        (username, email, full_name, hashed_password, now, now),
    )
    user_id = c.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": user_id,
        "username": username,
        "email": email,
        "full_name": full_name,
        "is_active": True,
        "is_superuser": False,
        "created_at": now,
        "updated_at": now,
    }


def update_user_last_login(user_id: int) -> None:
    """Update user's last login timestamp."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now(UTC).isoformat()
    c.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))
    conn.commit()
    conn.close()


# Token management functions
def save_refresh_token(user_id: int, token_hash: str, expires_at: datetime) -> None:
    """Save a refresh token for a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now(UTC).isoformat()
    c.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at, is_revoked)
           VALUES (?, ?, ?, ?, 0)""",
        (user_id, token_hash, expires_at.isoformat(), now),
    )
    conn.commit()
    conn.close()


def verify_refresh_token(token: str) -> int | None:
    """Verify a refresh token and return the user ID if valid."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute(
        """SELECT * FROM refresh_tokens
           WHERE token_hash = ? AND is_revoked = 0 AND expires_at > ?""",
        (token_hash, datetime.now(UTC).isoformat()),
    )
    row = c.fetchone()
    conn.close()

    if row:
        return row["user_id"]
    return None


def revoke_refresh_token(_token: str) -> None:
    """Revoke a refresh token."""
    # Note: In the current implementation, tokens are revoked on logout
    # This is handled by the auth API endpoint
    # Keeping this function for backward compatibility
