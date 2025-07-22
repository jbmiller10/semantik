#!/usr/bin/env python3
"""Test script to verify PostgreSQL migration compatibility."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

from alembic import command
from alembic.config import Config


def test_migration():
    """Test the migration against PostgreSQL."""

    # Get PostgreSQL connection URL from environment or use default
    postgres_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/semantik_test")

    print(f"Testing migration with PostgreSQL: {postgres_url}")

    # Create engine and test connection
    engine = create_engine(postgres_url)

    try:
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            print(f"PostgreSQL version: {result.scalar()}")

            # Clean up any existing enum types
            print("Cleaning up existing enum types...")
            conn.execute(text("DROP TYPE IF EXISTS document_status CASCADE"))
            conn.execute(text("DROP TYPE IF EXISTS permission_type CASCADE"))
            conn.execute(text("DROP TYPE IF EXISTS collection_status CASCADE"))
            conn.execute(text("DROP TYPE IF EXISTS operation_type CASCADE"))
            conn.execute(text("DROP TYPE IF EXISTS operation_status CASCADE"))
            conn.commit()

        # Set up Alembic configuration
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", postgres_url)

        # Run migrations
        print("\nRunning migrations...")
        command.upgrade(alembic_cfg, "head")
        print("✓ Migrations completed successfully!")

        # Verify enum types were created
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT typname 
                FROM pg_type 
                WHERE typname IN ('document_status', 'permission_type', 'collection_status', 'operation_type', 'operation_status')
                ORDER BY typname
            """
                )
            )
            enum_types = [row[0] for row in result]
            print(f"\nCreated enum types: {', '.join(enum_types)}")

        # Test downgrade
        print("\nTesting downgrade...")
        command.downgrade(alembic_cfg, "base")
        print("✓ Downgrade completed successfully!")

        # Verify enum types were dropped
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT typname 
                FROM pg_type 
                WHERE typname IN ('document_status', 'permission_type', 'collection_status', 'operation_type', 'operation_status')
            """
                )
            )
            remaining_types = [row[0] for row in result]
            if remaining_types:
                print(f"⚠️  Warning: Some enum types were not cleaned up: {', '.join(remaining_types)}")
            else:
                print("✓ All enum types cleaned up successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(test_migration())
