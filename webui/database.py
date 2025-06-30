#!/usr/bin/env python3
"""
Database access layer for Document Embedding Web UI
Centralizes all database operations for jobs, files, users, and tokens
"""

import hashlib
import logging
import os
import sqlite3
import sys
from datetime import datetime
from typing import Any

from passlib.context import CryptContext

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vecpipe.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = str(settings.WEBUI_DB)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Password hashing context for auth functions
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Database initialization and management
def init_db():
    """Initialize SQLite database for job tracking and authentication"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if tables exist and need migration
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    jobs_exists = c.fetchone() is not None

    if jobs_exists:
        # Check for missing columns and add them
        c.execute("PRAGMA table_info(jobs)")
        columns = [col[1] for col in c.fetchall()]

        if "vector_dim" not in columns:
            logger.info("Migrating database: adding vector_dim column")
            c.execute("ALTER TABLE jobs ADD COLUMN vector_dim INTEGER")

        if "quantization" not in columns:
            logger.info("Migrating database: adding quantization column")
            c.execute("ALTER TABLE jobs ADD COLUMN quantization TEXT DEFAULT 'float32'")

        if "instruction" not in columns:
            logger.info("Migrating database: adding instruction column")
            c.execute("ALTER TABLE jobs ADD COLUMN instruction TEXT")

        if "start_time" not in columns:
            logger.info("Migrating database: adding start_time column")
            c.execute("ALTER TABLE jobs ADD COLUMN start_time TEXT")

    # Check if files table exists and needs migration
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
    files_exists = c.fetchone() is not None

    if files_exists:
        # Check for missing columns and add them
        c.execute("PRAGMA table_info(files)")
        columns = [col[1] for col in c.fetchall()]

        if "doc_id" not in columns:
            logger.info("Migrating database: adding doc_id column to files table")
            c.execute("ALTER TABLE files ADD COLUMN doc_id TEXT")

    # Jobs table
    c.execute(
        """CREATE TABLE IF NOT EXISTS jobs
                 (id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  directory_path TEXT NOT NULL,
                  model_name TEXT NOT NULL,
                  chunk_size INTEGER,
                  chunk_overlap INTEGER,
                  batch_size INTEGER,
                  vector_dim INTEGER,
                  quantization TEXT,
                  instruction TEXT,
                  total_files INTEGER DEFAULT 0,
                  processed_files INTEGER DEFAULT 0,
                  failed_files INTEGER DEFAULT 0,
                  current_file TEXT,
                  start_time TEXT,
                  error TEXT)"""
    )

    # Files table
    c.execute(
        """CREATE TABLE IF NOT EXISTS files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  path TEXT NOT NULL,
                  size INTEGER NOT NULL,
                  modified TEXT NOT NULL,
                  extension TEXT NOT NULL,
                  hash TEXT,
                  doc_id TEXT,
                  status TEXT DEFAULT 'pending',
                  error TEXT,
                  chunks_created INTEGER DEFAULT 0,
                  vectors_created INTEGER DEFAULT 0,
                  FOREIGN KEY (job_id) REFERENCES jobs(id))"""
    )

    # Create indices
    c.execute("""CREATE INDEX IF NOT EXISTS idx_files_job_id ON files(job_id)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_files_doc_id ON files(doc_id)""")

    # Initialize auth tables
    init_auth_tables(conn, c)

    conn.commit()
    conn.close()


def init_auth_tables(conn: sqlite3.Connection, c: sqlite3.Cursor):
    """Initialize authentication tables in the database"""
    # Users table
    c.execute(
        """CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  full_name TEXT,
                  hashed_password TEXT NOT NULL,
                  is_active BOOLEAN DEFAULT 1,
                  created_at TEXT NOT NULL,
                  last_login TEXT)"""
    )

    # Refresh tokens table (for token revocation)
    c.execute(
        """CREATE TABLE IF NOT EXISTS refresh_tokens
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  token_hash TEXT UNIQUE NOT NULL,
                  expires_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  is_revoked BOOLEAN DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users(id))"""
    )

    # Create indices
    c.execute("""CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_refresh_tokens_hash ON refresh_tokens(token_hash)""")

    logger.info("Authentication tables initialized")


def reset_database():
    """Reset the database by dropping all tables and recreating them"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Drop tables
    c.execute("DROP TABLE IF EXISTS files")
    c.execute("DROP TABLE IF EXISTS jobs")
    c.execute("DROP TABLE IF EXISTS refresh_tokens")
    c.execute("DROP TABLE IF EXISTS users")

    conn.commit()
    conn.close()

    # Recreate tables
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
                  batch_size, vector_dim, quantization, instruction)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        ),
    )

    conn.commit()
    conn.close()

    return job_data["id"]


def update_job(job_id: str, updates: dict[str, Any]):
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
    values.append(datetime.utcnow().isoformat())

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


def list_jobs() -> list[dict[str, Any]]:
    """List all jobs"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

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


def delete_job(job_id: str):
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
def add_files_to_job(job_id: str, files: list[dict[str, Any]]):
    """Add multiple files to a job"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for file in files:
        # Generate doc_id as MD5 hash of file path
        doc_id = hashlib.md5(file["path"].encode()).hexdigest()[:16]

        c.execute(
            """INSERT INTO files (job_id, path, size, modified, extension, hash, doc_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, file["path"], file["size"], file["modified"], file["extension"], file.get("hash"), doc_id),
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
):
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
    created_at = datetime.utcnow().isoformat()

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


def update_user_last_login(user_id: int):
    """Update user's last login timestamp"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.utcnow().isoformat(), user_id))
    conn.commit()
    conn.close()


# Refresh token operations
def save_refresh_token(user_id: int, token_hash: str, expires_at: datetime):
    """Save refresh token to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at)
                 VALUES (?, ?, ?, ?)""",
        (user_id, token_hash, expires_at.isoformat(), datetime.utcnow().isoformat()),
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
        (datetime.utcnow().isoformat(),),
    )

    tokens = c.fetchall()
    conn.close()

    # Check each token
    for token_row in tokens:
        if pwd_context.verify(token, token_row["token_hash"]):
            return token_row["user_id"]

    return None


def revoke_refresh_token(token: str):
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


# Initialize database when module is imported
init_db()
