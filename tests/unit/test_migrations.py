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

import pytest
from sqlalchemy import create_engine, inspect


class TestMigrations:
    """Test database migration functionality."""

    @pytest.fixture()
    def migration_db(self, tmp_path):
        """Create a temporary database for migration testing."""
        db_path = tmp_path / "migration_test.db"
        return str(db_path)

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

        # Expected tables for collections-based schema
        expected_tables = {
            "alembic_version",
            "users",
            "refresh_tokens",
            "collections",
            "documents",
            "api_keys",
            "collection_permissions",
            "collection_sources",
            "operations",
            "collection_audit_log",
            "collection_resource_limits",
            "operation_metrics",
        }
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

        # Insert user
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password, created_at, is_active, is_superuser) VALUES (?, ?, ?, ?, ?, ?)",
            ("testuser", "test@example.com", "hashedpw", "2023-01-01T00:00:00", 1, 0),
        )
        user_id = cursor.lastrowid

        # Insert collection
        cursor.execute(
            """INSERT INTO collections (id, name, description, owner_id, vector_store_name,
               embedding_model, chunk_size, chunk_overlap, is_public, created_at, updated_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "coll1",
                "Test Collection",
                "Description",
                user_id,
                "vec_store_1",
                "model1",
                1000,
                200,
                0,
                "2023-01-01T00:00:00",
                "2023-01-01T00:00:00",
                "ready",
            ),
        )
        conn.commit()

        # Verify data exists
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM collections")
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

        cursor.execute("SELECT name FROM collections WHERE id = 'coll1'")
        assert cursor.fetchone()[0] == "Test Collection"

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
        assert "collections" in table_names
        assert "documents" in table_names
        assert "operations" in table_names
        assert "users" in table_names
        assert "refresh_tokens" in table_names
        assert "api_keys" in table_names
        assert "collection_permissions" in table_names

        # Verify indexes exist on collections
        collections_indexes = {idx["name"] for idx in inspector.get_indexes("collections")}
        expected_indexes = {"ix_collections_name", "ix_collections_owner_id", "ix_collections_is_public"}
        assert expected_indexes.issubset(
            collections_indexes
        ), f"Missing indexes: {expected_indexes - collections_indexes}"

        # Verify indexes exist on documents
        documents_indexes = {idx["name"] for idx in inspector.get_indexes("documents")}
        expected_doc_indexes = {
            "ix_documents_collection_id",
            "ix_documents_content_hash",
            "ix_documents_status",
            "ix_documents_collection_content_hash",
        }
        assert expected_doc_indexes.issubset(
            documents_indexes
        ), f"Missing indexes: {expected_doc_indexes - documents_indexes}"

        engine.dispose()
