#!/usr/bin/env python3
"""
Backup database before running migrations.

This script creates a timestamped backup of the database before applying migrations,
providing a safety net for production deployments.
"""

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def backup_database(db_path: str) -> str:
    """Create a timestamped backup of the database."""
    db_file = Path(db_path)

    if not db_file.exists():
        print(f"Database file not found: {db_path}")
        return ""

    # Create backup filename with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_name = f"{db_file.stem}_backup_{timestamp}{db_file.suffix}"
    backup_path = db_file.parent / backup_name

    # Copy database file
    try:
        shutil.copy2(db_file, backup_path)
        print(f"Database backed up to: {backup_path}")
        return str(backup_path)
    except Exception as e:
        print(f"Failed to backup database: {e}")
        sys.exit(1)


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python backup_before_migration.py <database_path>")
        sys.exit(1)

    db_path = sys.argv[1]
    backup_path = backup_database(db_path)

    if backup_path:
        print(f"Backup completed successfully: {backup_path}")
        print("You can now safely run migrations.")
        print(f"To restore: cp {backup_path} {db_path}")


if __name__ == "__main__":
    main()
