#!/usr/bin/env python3
"""
SQLite database implementation for shared package.
Centralizes all database operations for jobs, files, users, and tokens.

This module contains the actual database implementation that was previously
in webui/database.py. It's now part of the shared package to enable proper
service decoupling.
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

    # Run alembic upgrade head
    logger.info("Running database migrations with Alembic...")
    try:
        # Need to set PYTHONPATH for alembic to find our packages
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root / "packages") + ":" + env.get("PYTHONPATH", "")
        # Pass the database path to alembic via environment variable
        env["ALEMBIC_DATABASE_URL"] = f"sqlite:///{DB_PATH}"

        # Change to project root directory to find alembic.ini
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        if result.stdout:
            logger.info(f"Alembic output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Alembic stderr: {result.stderr}")
        logger.info("Database migrations completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run database migrations: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Database migration failed: {e.stderr}") from e


def init_auth_tables(_conn: sqlite3.Connection, _c: sqlite3.Cursor) -> None:
    """DEPRECATED: Authentication tables are now created via Alembic migrations"""
    # This function is kept for backward compatibility but does nothing
    # All table creation is handled by Alembic migrations


def reset_database() -> None:
    """Reset the database by dropping all tables and recreating them using Alembic"""
    import subprocess
    import sys
    from contextlib import suppress
    from pathlib import Path

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Drop tables
    c.execute("DROP TABLE IF EXISTS files")
    c.execute("DROP TABLE IF EXISTS jobs")
    c.execute("DROP TABLE IF EXISTS refresh_tokens")
    c.execute("DROP TABLE IF EXISTS users")

    conn.commit()
    conn.close()

    # Find the project root (where alembic.ini is located)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent

    # Downgrade to nothing (remove all migrations)
    logger.info("Running database downgrade...")
    with suppress(subprocess.CalledProcessError):
        # It's OK if downgrade fails (e.g., if alembic_version table doesn't exist)
        subprocess.run(
            [sys.executable, "-m", "alembic", "downgrade", "base"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )

    # Recreate tables using Alembic
    init_db()
    logger.info("Database reset successfully")


def get_database_stats() -> dict[str, Any]:
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get counts
    c.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM jobs WHERE status = 'completed'")
    completed_jobs = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM jobs WHERE status = 'failed'")
    failed_jobs = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
    running_jobs = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM files")
    total_files = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    conn.close()

    return {
        "jobs": {"total": total_jobs, "completed": completed_jobs, "failed": failed_jobs, "running": running_jobs},
        "files": {"total": total_files},
        "users": {"total": total_users},
    }


# Job-related database operations
def create_job(job_data: dict[str, Any]) -> str:
    """Create a new job in the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """INSERT INTO jobs
                 (id, name, description, status, created_at, updated_at,
                  directory_path, model_name, chunk_size, chunk_overlap,
                  batch_size, vector_dim, quantization, instruction, user_id,
                  parent_job_id, mode)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_data["id"],
            job_data["name"],
            job_data["description"],
            job_data["status"],
            job_data["created_at"],
            job_data["updated_at"],
            job_data["directory_path"],
            job_data["model_name"],
            job_data["chunk_size"],
            job_data["chunk_overlap"],
            job_data["batch_size"],
            job_data.get("vector_dim"),
            job_data.get("quantization", "float32"),
            job_data.get("instruction"),
            job_data.get("user_id"),
            job_data.get("parent_job_id"),
            job_data.get("mode", "create"),
        ),
    )

    conn.commit()
    conn.close()

    return str(job_data["id"])


def update_job(job_id: str, updates: dict[str, Any]) -> None:
    """Update job information"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Build update query dynamically
    update_fields = []
    values = []
    for key, value in updates.items():
        update_fields.append(f"{key} = ?")
        values.append(value)

    # Always update the updated_at timestamp
    update_fields.append("updated_at = ?")
    values.append(datetime.now(UTC).isoformat())

    # Add job_id for WHERE clause
    values.append(job_id)

    query = f"UPDATE jobs SET {', '.join(update_fields)} WHERE id = ?"
    c.execute(query, values)

    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict[str, Any] | None:
    """Get job by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job = c.fetchone()

    if job:
        job_dict = dict(job)
        # Get file statistics
        c.execute("SELECT COUNT(*) FROM files WHERE job_id = ? AND status = 'completed'", (job_id,))
        job_dict["completed_files"] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM files WHERE job_id = ? AND status = 'failed'", (job_id,))
        job_dict["failed_files"] = c.fetchone()[0]

    conn.close()

    return job_dict if job else None


def list_jobs(user_id: int | None = None) -> list[dict[str, Any]]:
    """List all jobs, optionally filtered by user"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if user_id is not None:
        # Show jobs owned by user OR jobs without user_id (legacy jobs)
        c.execute("SELECT * FROM jobs WHERE user_id = ? OR user_id IS NULL ORDER BY created_at DESC", (user_id,))
    else:
        c.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    jobs = c.fetchall()

    result = []
    for job in jobs:
        job_dict = dict(job)
        # Get file statistics
        c.execute("SELECT COUNT(*) FROM files WHERE job_id = ?", (job["id"],))
        job_dict["total_files"] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM files WHERE job_id = ? AND status = 'completed'", (job["id"],))
        job_dict["completed_files"] = c.fetchone()[0]

        result.append(job_dict)

    conn.close()

    return result


def delete_job(job_id: str) -> None:
    """Delete a job and its associated files"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Delete associated files first
    c.execute("DELETE FROM files WHERE job_id = ?", (job_id,))

    # Delete the job
    c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    conn.commit()
    conn.close()


# File-related database operations
def add_files_to_job(job_id: str, files: list[dict[str, Any]]) -> None:
    """Add multiple files to a job"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for file in files:
        # Generate doc_id as MD5 hash of file path
        doc_id = hashlib.md5(file["path"].encode()).hexdigest()[:16]

        c.execute(
            """INSERT INTO files (job_id, path, size, modified, extension, hash, doc_id, content_hash)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                file["path"],
                file["size"],
                file["modified"],
                file["extension"],
                file.get("hash"),
                doc_id,
                file.get("content_hash"),
            ),
        )

    # Update job total files count
    c.execute("UPDATE jobs SET total_files = ? WHERE id = ?", (len(files), job_id))

    conn.commit()
    conn.close()


def update_file_status(
    job_id: str,
    file_path: str,
    status: str,
    error: str | None = None,
    chunks_created: int = 0,
    vectors_created: int = 0,
) -> None:
    """Update file processing status"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """UPDATE files
                 SET status = ?, error = ?, chunks_created = ?, vectors_created = ?
                 WHERE job_id = ? AND path = ?""",
        (status, error, chunks_created, vectors_created, job_id, file_path),
    )

    conn.commit()
    conn.close()


def get_job_files(job_id: str, status: str | None = None) -> list[dict[str, Any]]:
    """Get files for a job, optionally filtered by status"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if status:
        c.execute("SELECT * FROM files WHERE job_id = ? AND status = ?", (job_id, status))
    else:
        c.execute("SELECT * FROM files WHERE job_id = ?", (job_id,))

    files = c.fetchall()
    conn.close()

    return [dict(file) for file in files]


def get_job_total_vectors(job_id: str) -> int:
    """Get total vectors created for a job"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """SELECT COALESCE(SUM(vectors_created), 0) as total
           FROM files
           WHERE job_id = ? AND status = 'completed'""",
        (job_id,),
    )

    result = c.fetchone()
    conn.close()

    return int(result[0]) if result else 0


def get_duplicate_files_in_collection(collection_name: str, content_hashes: list[str]) -> set[str]:
    """Check which content hashes already exist in a collection"""
    if not content_hashes:
        return set()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Find all jobs for this collection
    c.execute("""SELECT id FROM jobs WHERE name = ? AND status = 'completed'""", (collection_name,))
    job_ids = [row[0] for row in c.fetchall()]

    if not job_ids:
        conn.close()
        return set()

    # Check which content hashes exist in these jobs
    placeholders = ",".join("?" * len(content_hashes))
    job_placeholders = ",".join("?" * len(job_ids))

    query = f"""SELECT DISTINCT content_hash
                FROM files
                WHERE job_id IN ({job_placeholders})
                AND content_hash IN ({placeholders})
                AND content_hash IS NOT NULL"""

    c.execute(query, job_ids + content_hashes)

    existing_hashes = {row[0] for row in c.fetchall()}
    conn.close()

    return existing_hashes


def get_collection_metadata(collection_name: str) -> dict[str, Any] | None:
    """Get the metadata from the first (parent) job of a collection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Find the original job (mode='create' or no parent_job_id)
    c.execute(
        """SELECT * FROM jobs
           WHERE name = ?
           AND (mode = 'create' OR parent_job_id IS NULL)
           ORDER BY created_at ASC
           LIMIT 1""",
        (collection_name,),
    )

    job = c.fetchone()
    conn.close()

    return dict(job) if job else None


def list_collections(user_id: int | None = None) -> list[dict[str, Any]]:
    """Get unique collections with aggregated stats"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if user_id is not None:
        # Show collections where user owns at least one job OR legacy collections without user_id
        query = """
            SELECT
                name,
                COUNT(DISTINCT id) as job_count,
                SUM(total_files) as total_files,
                MAX(created_at) as created_at,
                MAX(updated_at) as updated_at,
                MIN(CASE WHEN mode = 'create' OR parent_job_id IS NULL THEN model_name END) as model_name
            FROM jobs
            WHERE status != 'failed'
            AND (user_id = ? OR user_id IS NULL)
            GROUP BY name
            ORDER BY MAX(updated_at) DESC
        """
        c.execute(query, (user_id,))
    else:
        query = """
            SELECT
                name,
                COUNT(DISTINCT id) as job_count,
                SUM(total_files) as total_files,
                MAX(created_at) as created_at,
                MAX(updated_at) as updated_at,
                MIN(CASE WHEN mode = 'create' OR parent_job_id IS NULL THEN model_name END) as model_name
            FROM jobs
            WHERE status != 'failed'
            GROUP BY name
            ORDER BY MAX(updated_at) DESC
        """
        c.execute(query)

    collections = c.fetchall()
    result = []

    for collection in collections:
        collection_dict = dict(collection)
        # Get total vectors for all jobs in this collection
        c.execute(
            """SELECT id FROM jobs WHERE name = ? AND status = 'completed'""",
            (collection["name"],),
        )
        job_ids = [row[0] for row in c.fetchall()]

        total_vectors = 0
        for job_id in job_ids:
            c.execute(
                """SELECT COALESCE(SUM(vectors_created), 0) as total
                   FROM files
                   WHERE job_id = ? AND status = 'completed'""",
                (job_id,),
            )
            result_row = c.fetchone()
            total_vectors += result_row[0] if result_row else 0

        collection_dict["total_vectors"] = total_vectors
        result.append(collection_dict)

    conn.close()
    return result


def get_collection_details(collection_name: str, user_id: int) -> dict[str, Any] | None:
    """Get detailed info for a single collection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Check if user has access to this collection
    c.execute(
        """SELECT COUNT(*) FROM jobs
           WHERE name = ? AND (user_id = ? OR user_id IS NULL)""",
        (collection_name, user_id),
    )
    if c.fetchone()[0] == 0:
        conn.close()
        return None

    # Get metadata from parent job
    metadata = get_collection_metadata(collection_name)
    if not metadata:
        conn.close()
        return None

    # Get all jobs for this collection
    c.execute(
        """SELECT id, status, created_at, updated_at, directory_path,
                  total_files, processed_files, failed_files, mode
           FROM jobs
           WHERE name = ?
           ORDER BY created_at DESC""",
        (collection_name,),
    )
    jobs = [dict(job) for job in c.fetchall()]

    # Get unique source directories
    c.execute(
        """SELECT DISTINCT directory_path
           FROM jobs
           WHERE name = ?""",
        (collection_name,),
    )
    source_directories = [row[0] for row in c.fetchall()]

    # Calculate total stats
    total_files = sum(job.get("total_files", 0) for job in jobs)
    total_vectors = 0
    total_size = 0

    for job in jobs:
        if job["status"] == "completed":
            # Get vectors for this job
            c.execute(
                """SELECT COALESCE(SUM(vectors_created), 0) as total,
                          COALESCE(SUM(size), 0) as total_size
                   FROM files
                   WHERE job_id = ? AND status = 'completed'""",
                (job["id"],),
            )
            result_row = c.fetchone()
            if result_row:
                total_vectors += result_row[0] or 0
                total_size += result_row[1] or 0

    conn.close()

    return {
        "name": collection_name,
        "stats": {
            "total_files": total_files,
            "total_vectors": total_vectors,
            "total_size": total_size,
            "job_count": len(jobs),
        },
        "configuration": {
            "model_name": metadata["model_name"],
            "chunk_size": metadata["chunk_size"],
            "chunk_overlap": metadata["chunk_overlap"],
            "quantization": metadata.get("quantization", "float32"),
            "vector_dim": metadata.get("vector_dim"),
            "instruction": metadata.get("instruction"),
        },
        "source_directories": source_directories,
        "jobs": jobs,
    }


def get_collection_files(collection_name: str, user_id: int, page: int = 1, limit: int = 50) -> dict[str, Any]:
    """Get paginated files for a collection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Check access
    c.execute(
        """SELECT id FROM jobs
           WHERE name = ? AND (user_id = ? OR user_id IS NULL)""",
        (collection_name, user_id),
    )
    job_ids = [row[0] for row in c.fetchall()]

    if not job_ids:
        conn.close()
        return {"files": [], "total": 0, "page": page, "pages": 0}

    # Count total files
    placeholders = ",".join("?" * len(job_ids))
    c.execute(
        f"""SELECT COUNT(*) FROM files
            WHERE job_id IN ({placeholders})""",
        job_ids,
    )
    total = c.fetchone()[0]

    # Calculate pagination
    offset = (page - 1) * limit
    pages = (total + limit - 1) // limit

    # Get paginated files
    c.execute(
        f"""SELECT f.*, j.name as collection_name
            FROM files f
            JOIN jobs j ON f.job_id = j.id
            WHERE f.job_id IN ({placeholders})
            ORDER BY f.path
            LIMIT ? OFFSET ?""",
        job_ids + [limit, offset],
    )

    files = [dict(file) for file in c.fetchall()]
    conn.close()

    return {"files": files, "total": total, "page": page, "pages": pages}


def rename_collection(old_name: str, new_name: str, user_id: int) -> bool:
    """Rename collection display name"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # Check if user owns at least one job in the collection
        c.execute(
            """SELECT COUNT(*) FROM jobs
               WHERE name = ? AND (user_id = ? OR user_id IS NULL)""",
            (old_name, user_id),
        )
        if c.fetchone()[0] == 0:
            conn.close()
            return False

        # Check if new name already exists
        c.execute("SELECT COUNT(*) FROM jobs WHERE name = ?", (new_name,))
        if c.fetchone()[0] > 0:
            conn.close()
            return False

        # Update all jobs with the old name
        c.execute(
            """UPDATE jobs
               SET name = ?, updated_at = ?
               WHERE name = ?""",
            (new_name, datetime.now(UTC).isoformat(), old_name),
        )

        conn.commit()
        conn.close()
        return True

    except Exception:
        conn.rollback()
        conn.close()
        return False


def delete_collection(collection_name: str, user_id: int) -> dict[str, Any]:
    """Delete collection and all associated data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get all job IDs for this collection that user has access to
    c.execute(
        """SELECT id FROM jobs
           WHERE name = ? AND (user_id = ? OR user_id IS NULL)""",
        (collection_name, user_id),
    )
    job_ids = [row[0] for row in c.fetchall()]

    if not job_ids:
        conn.close()
        return {"job_ids": [], "qdrant_collections": []}

    # Get Qdrant collection names
    qdrant_collections = [f"job_{job_id}" for job_id in job_ids]

    # Delete files first
    placeholders = ",".join("?" * len(job_ids))
    c.execute(f"DELETE FROM files WHERE job_id IN ({placeholders})", job_ids)

    # Delete jobs
    c.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", job_ids)

    conn.commit()
    conn.close()

    return {"job_ids": job_ids, "qdrant_collections": qdrant_collections}


# User-related database operations
def get_user(username: str) -> dict[str, Any] | None:
    """Get user by username"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()

    return dict(user) if user else None


# Removed unused function: get_user_by_email
# This function was defined but never called in the codebase


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Get user by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()

    return dict(user) if user else None


def create_user(username: str, email: str, hashed_password: str, full_name: str | None = None) -> dict[str, Any]:
    """Create a new user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if user already exists
    c.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    if c.fetchone():
        conn.close()
        raise ValueError("User with this username or email already exists")

    # Create user
    created_at = datetime.now(UTC).isoformat()

    c.execute(
        """INSERT INTO users (username, email, full_name, hashed_password, created_at)
                 VALUES (?, ?, ?, ?, ?)""",
        (username, email, full_name, hashed_password, created_at),
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
        "created_at": created_at,
        "last_login": None,
    }


def update_user_last_login(user_id: int) -> None:
    """Update user's last login timestamp"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(UTC).isoformat(), user_id))
    conn.commit()
    conn.close()


# Refresh token operations
def save_refresh_token(user_id: int, token_hash: str, expires_at: datetime) -> None:
    """Save refresh token to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at)
                 VALUES (?, ?, ?, ?)""",
        (user_id, token_hash, expires_at.isoformat(), datetime.now(UTC).isoformat()),
    )

    conn.commit()
    conn.close()


def verify_refresh_token(token: str) -> int | None:
    """Verify refresh token and return user_id if valid"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get all non-revoked, non-expired tokens
    c.execute(
        """SELECT * FROM refresh_tokens
                 WHERE is_revoked = 0 AND expires_at > ?""",
        (datetime.now(UTC).isoformat(),),
    )

    tokens = c.fetchall()
    conn.close()

    # Check each token
    for token_row in tokens:
        if pwd_context.verify(token, token_row["token_hash"]):
            return int(token_row["user_id"])

    return None


def revoke_refresh_token(_token: str) -> None:
    """Revoke a refresh token"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Find and revoke the token
    c.execute(
        """UPDATE refresh_tokens SET is_revoked = 1
                 WHERE token_hash IN (SELECT token_hash FROM refresh_tokens WHERE is_revoked = 0)"""
    )

    conn.commit()
    conn.close()


# Database initialization is now done explicitly when needed
# Previously init_db() was called here, but this caused circular dependencies with Alembic
