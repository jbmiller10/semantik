"""Utility script for managing migration backups.

This script provides functions to:
1. Check for existing backups
2. Clean up old backups past retention period
3. Verify backup integrity
4. Show migration history
"""

import argparse
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackupManager:
    """Manages database migration backups."""

    def __init__(self, database_url: str):
        """Initialize backup manager with database connection."""
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

    def list_backups(self) -> list[dict]:
        """List all migration backups."""
        with self.Session() as session:
            result = session.execute(
                text(
                    """
                    SELECT
                        id,
                        backup_table_name,
                        original_table_name,
                        record_count,
                        created_at,
                        migration_revision,
                        retention_until,
                        CASE
                            WHEN retention_until < NOW() THEN 'EXPIRED'
                            WHEN retention_until < NOW() + INTERVAL '1 day' THEN 'EXPIRING SOON'
                            ELSE 'ACTIVE'
                        END as status
                    FROM migration_backups
                    ORDER BY created_at DESC
                    """
                )
            )

            backups = []
            for row in result:
                backups.append(
                    {
                        "id": row[0],
                        "backup_table": row[1],
                        "original_table": row[2],
                        "record_count": row[3],
                        "created_at": row[4],
                        "migration_revision": row[5],
                        "retention_until": row[6],
                        "status": row[7],
                    }
                )

            return backups

    def cleanup_expired_backups(self, dry_run: bool = True) -> int:
        """Clean up backups past their retention period."""
        with self.Session() as session:
            # Find expired backups
            result = session.execute(
                text(
                    """
                    SELECT backup_table_name
                    FROM migration_backups
                    WHERE retention_until < NOW()
                    AND backup_table_name NOT LIKE '%REMINDER%'
                    """
                )
            )

            expired_tables = [row[0] for row in result]

            if not expired_tables:
                logger.info("No expired backups found")
                return 0

            logger.info(f"Found {len(expired_tables)} expired backup(s)")

            if dry_run:
                logger.info("DRY RUN - Would delete the following tables:")
                for table in expired_tables:
                    logger.info(f"  - {table}")
                return len(expired_tables)

            # Actually delete expired backups
            deleted_count = 0
            for table_name in expired_tables:
                try:
                    # Check if table exists
                    exists_result = session.execute(
                        text(
                            """
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_schema = 'public'
                                AND table_name = :table_name
                            )
                            """
                        ),
                        {"table_name": table_name},
                    )

                    if exists_result.scalar():
                        session.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                        logger.info(f"Deleted backup table: {table_name}")

                    # Remove from tracking table
                    session.execute(
                        text(
                            """
                            DELETE FROM migration_backups
                            WHERE backup_table_name = :table_name
                            """
                        ),
                        {"table_name": table_name},
                    )

                    deleted_count += 1

                except Exception as e:
                    logger.error(f"Error deleting {table_name}: {e}")
                    continue

            session.commit()
            logger.info(f"Cleaned up {deleted_count} expired backup(s)")
            return deleted_count

    def verify_backup(self, backup_table_name: str) -> bool:
        """Verify a specific backup table's integrity."""
        with self.Session() as session:
            try:
                # Check if backup table exists
                exists_result = session.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = :table_name
                        )
                        """
                    ),
                    {"table_name": backup_table_name},
                )

                if not exists_result.scalar():
                    logger.error(f"Backup table {backup_table_name} does not exist")
                    return False

                # Get record count from backup
                count_result = session.execute(text(f"SELECT COUNT(*) FROM {backup_table_name}"))
                actual_count = count_result.scalar()

                # Get expected count from tracking table
                tracking_result = session.execute(
                    text(
                        """
                        SELECT record_count
                        FROM migration_backups
                        WHERE backup_table_name = :table_name
                        """
                    ),
                    {"table_name": backup_table_name},
                )

                expected_count = tracking_result.scalar()

                if expected_count is None:
                    logger.warning(f"No tracking entry found for {backup_table_name}")
                    return True

                if actual_count != expected_count:
                    logger.error(
                        f"Count mismatch for {backup_table_name}: expected {expected_count}, found {actual_count}"
                    )
                    return False

                logger.info(f"Backup {backup_table_name} verified: {actual_count} records")
                return True

            except Exception as e:
                logger.error(f"Error verifying {backup_table_name}: {e}")
                return False

    def extend_retention(self, backup_table_name: str, days: int = 7) -> bool:
        """Extend the retention period for a backup."""
        with self.Session() as session:
            try:
                session.execute(
                    text(
                        """
                        UPDATE migration_backups
                        SET retention_until = NOW() + INTERVAL :days
                        WHERE backup_table_name = :table_name
                        """
                    ),
                    {"days": f"{days} days", "table_name": backup_table_name},
                )

                session.commit()
                logger.info(f"Extended retention for {backup_table_name} by {days} days")
                return True

            except Exception as e:
                logger.error(f"Error extending retention: {e}")
                return False

    def show_migration_status(self):
        """Show current migration status and chunk distribution."""
        with self.Session() as session:
            try:
                # Check if chunks table exists
                exists_result = session.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = 'chunks'
                        )
                        """
                    )
                )

                if not exists_result.scalar():
                    logger.info("Chunks table does not exist")
                    return

                # Get total chunks
                count_result = session.execute(text("SELECT COUNT(*) FROM chunks"))
                total_chunks = count_result.scalar()

                logger.info(f"Total chunks: {total_chunks}")

                # Check partition distribution
                dist_result = session.execute(
                    text(
                        """
                        SELECT
                            COUNT(DISTINCT partition_key) as partitions_used,
                            AVG(cnt) as avg_per_partition,
                            MAX(cnt) as max_per_partition,
                            MIN(cnt) as min_per_partition
                        FROM (
                            SELECTpartition_key, COUNT(*) as cnt
                            FROM chunks
                            GROUP BY partition_key
                        ) sub
                        """
                    )
                )

                row = dist_result.fetchone()
                if row:
                    logger.info(f"Partitions used: {row[0]}/100")
                    logger.info(f"Average chunks per partition: {row[1]:.1f}")
                    logger.info(f"Max chunks in partition: {row[2]}")
                    logger.info(f"Min chunks in partition: {row[3]}")

            except Exception as e:
                logger.error(f"Error checking migration status: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Manage migration backups")
    parser.add_argument("--database-url", required=True, help="PostgreSQL database URL")
    parser.add_argument(
        "--action", choices=["list", "cleanup", "verify", "extend", "status"], required=True, help="Action to perform"
    )
    parser.add_argument("--table-name", help="Backup table name (for verify/extend actions)")
    parser.add_argument("--days", type=int, default=7, help="Days to extend retention (for extend action)")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run (for cleanup action)")

    args = parser.parse_args()

    manager = BackupManager(args.database_url)

    if args.action == "list":
        backups = manager.list_backups()
        if backups:
            print("\nMigration Backups:")
            print("-" * 80)
            for backup in backups:
                print(f"Table: {backup['backup_table']}")
                print(f"  Original: {backup['original_table']}")
                print(f"  Records: {backup['record_count']}")
                print(f"  Created: {backup['created_at']}")
                print(f"  Expires: {backup['retention_until']}")
                print(f"  Status: {backup['status']}")
                print()
        else:
            print("No backups found")

    elif args.action == "cleanup":
        deleted = manager.cleanup_expired_backups(dry_run=args.dry_run)
        if args.dry_run:
            print(f"\nDry run: Would delete {deleted} expired backup(s)")
        else:
            print(f"\nDeleted {deleted} expired backup(s)")

    elif args.action == "verify":
        if not args.table_name:
            print("Error: --table-name required for verify action")
            return

        if manager.verify_backup(args.table_name):
            print(f"Backup {args.table_name} is valid")
        else:
            print(f"Backup {args.table_name} verification failed")

    elif args.action == "extend":
        if not args.table_name:
            print("Error: --table-name required for extend action")
            return

        if manager.extend_retention(args.table_name, args.days):
            print(f"Extended retention for {args.table_name} by {args.days} days")
        else:
            print(f"Failed to extend retention for {args.table_name}")

    elif args.action == "status":
        manager.show_migration_status()


if __name__ == "__main__":
    main()
