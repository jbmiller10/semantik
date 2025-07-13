#!/usr/bin/env python3
"""
Unit tests for Alembic database migrations.

This module tests that migrations work correctly, including:
- Upgrade/downgrade cycles
- Schema integrity after migrations
- Data preservation during migrations
"""

import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect


class TestMigrations:
    """Test database migration functionality."""

    @pytest.fixture
    def migration_db(self, tmp_path):
        """Create a temporary database for migration testing."""
        db_path = tmp_path / "migration_test.db"
        yield str(db_path)

    def run_alembic_command(self, command: str, db_path: str) -> tuple[int, str, str]:
        """Run an alembic command with the test database."""
        project_root = Path(__file__).parent.parent.parent
        env = {
            "PYTHONPATH": str(project_root / "packages"),
            "ALEMBIC_DATABASE_URL": f"sqlite:///{db_path}",
        }
        
        result = subprocess.run(
            [sys.executable, "-m", "alembic"] + command.split(),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            env=env,
        )
        
        return result.returncode, result.stdout, result.stderr

    def test_initial_migration_creates_all_tables(self, migration_db):
        """Test that the initial migration creates all expected tables."""
        # Run upgrade to head
        returncode, stdout, stderr = self.run_alembic_command("upgrade head", migration_db)
        assert returncode == 0, f"Migration failed: {stderr}"
        
        # Verify all tables exist
        conn = sqlite3.connect(migration_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in cursor.fetchall()}
        
        expected_tables = {"alembic_version", "files", "jobs", "refresh_tokens", "users"}
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"
        
        conn.close()

    def test_migration_is_idempotent(self, migration_db):
        """Test that running the migration twice doesn't cause errors."""
        # First upgrade
        returncode1, _, stderr1 = self.run_alembic_command("upgrade head", migration_db)
        assert returncode1 == 0, f"First migration failed: {stderr1}"
        
        # Second upgrade (should be no-op)
        returncode2, stdout2, stderr2 = self.run_alembic_command("upgrade head", migration_db)
        assert returncode2 == 0, f"Second migration failed: {stderr2}"
        assert "Running upgrade" not in stdout2, "Migration ran again when it shouldn't have"

    def test_downgrade_removes_all_tables(self, migration_db):
        """Test that downgrade properly removes all tables."""
        # First upgrade
        returncode, _, stderr = self.run_alembic_command("upgrade head", migration_db)
        assert returncode == 0, f"Upgrade failed: {stderr}"
        
        # Then downgrade
        returncode, _, stderr = self.run_alembic_command("downgrade base", migration_db)
        assert returncode == 0, f"Downgrade failed: {stderr}"
        
        # Verify tables are gone (except alembic_version)
        conn = sqlite3.connect(migration_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'alembic_version'")
        tables = cursor.fetchall()
        assert len(tables) == 0, f"Tables still exist after downgrade: {tables}"
        
        conn.close()

    def test_migration_preserves_existing_data(self, migration_db):
        """Test that migrations preserve existing data."""
        # Create initial schema and add data
        returncode, _, stderr = self.run_alembic_command("upgrade head", migration_db)
        assert returncode == 0, f"Initial upgrade failed: {stderr}"
        
        # Insert test data
        conn = sqlite3.connect(migration_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password, created_at) VALUES (?, ?, ?, ?)",
            ("testuser", "test@example.com", "hashedpw", "2023-01-01T00:00:00"),
        )
        cursor.execute(
            "INSERT INTO jobs (id, name, status, created_at, updated_at, directory_path, model_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("job1", "Test Job", "completed", "2023-01-01T00:00:00", "2023-01-01T00:00:00", "/test", "model1"),
        )
        conn.commit()
        
        # Verify data exists
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM jobs")
        assert cursor.fetchone()[0] == 1
        
        conn.close()
        
        # Run current migration again (should be no-op but shouldn't lose data)
        returncode, _, stderr = self.run_alembic_command("upgrade head", migration_db)
        assert returncode == 0, f"Re-upgrade failed: {stderr}"
        
        # Verify data still exists
        conn = sqlite3.connect(migration_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT username FROM users WHERE username = 'testuser'")
        assert cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM jobs WHERE id = 'job1'")
        assert cursor.fetchone()[0] == "Test Job"
        
        conn.close()

    def test_schema_matches_models(self, migration_db):
        """Test that the migrated schema matches the SQLAlchemy models."""
        # Run migration
        returncode, _, stderr = self.run_alembic_command("upgrade head", migration_db)
        assert returncode == 0, f"Migration failed: {stderr}"
        
        # Create engine and inspect schema
        engine = create_engine(f"sqlite:///{migration_db}")
        inspector = inspect(engine)
        
        # Check that all expected tables exist
        table_names = inspector.get_table_names()
        assert "jobs" in table_names
        assert "files" in table_names
        assert "users" in table_names
        assert "refresh_tokens" in table_names
        
        # Verify indexes exist
        files_indexes = {idx['name'] for idx in inspector.get_indexes('files')}
        expected_indexes = {
            'idx_files_job_id',
            'idx_files_status', 
            'idx_files_doc_id',
            'idx_files_content_hash',
            'idx_files_job_content_hash'
        }
        assert expected_indexes.issubset(files_indexes), f"Missing indexes: {expected_indexes - files_indexes}"
        
        engine.dispose()