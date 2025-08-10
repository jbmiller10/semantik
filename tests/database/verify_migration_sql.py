#!/usr/bin/env python3
"""
Verify the 100-partition migration SQL is valid by extracting and displaying it.
This doesn't require a database connection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alembic.versions.ae558c9e183f_implement_100_direct_list_partitions import upgrade


def extract_sql_statements():
    """Extract SQL statements from the migration."""

    class MockConnection:
        """Mock connection to capture SQL statements."""
        def __init__(self):
            self.statements = []

        def execute(self, statement):
            # Extract SQL text
            if hasattr(statement, 'text'):
                sql = str(statement.text)
            else:
                sql = str(statement)
            self.statements.append(sql)

    class MockOp:
        """Mock Alembic op to capture operations."""
        def __init__(self):
            self.conn = MockConnection()

        def get_bind(self):
            return self.conn

    # Monkey-patch the op module
    import alembic.op as real_op
    mock_op = MockOp()

    # Temporarily replace op functions
    original_get_bind = real_op.get_bind
    real_op.get_bind = lambda: mock_op.conn

    try:
        # Call upgrade function
        upgrade()

        print("=" * 80)
        print("MIGRATION SQL STATEMENTS - UPGRADE")
        print("=" * 80)

        for i, stmt in enumerate(mock_op.conn.statements, 1):
            print(f"\n-- Statement {i}")
            print(stmt[:500] if len(stmt) > 500 else stmt)  # Truncate very long statements

        print("\n" + "=" * 80)
        print(f"Total statements: {len(mock_op.conn.statements)}")

        # Verify key operations
        print("\n" + "=" * 80)
        print("VERIFICATION CHECKS")
        print("=" * 80)

        checks = {
            "DROP old chunks table": any("DROP TABLE IF EXISTS chunks" in s for s in mock_op.conn.statements),
            "CREATE new chunks table": any("CREATE TABLE chunks" in s for s in mock_op.conn.statements),
            "PARTITION BY LIST": any("PARTITION BY LIST" in s for s in mock_op.conn.statements),
            "Uses hashtext function": any("hashtext" in s for s in mock_op.conn.statements),
            "Creates 100 partitions": any("FOR i IN 0..99" in s for s in mock_op.conn.statements),
            "Creates monitoring views": any("partition_health" in s for s in mock_op.conn.statements),
            "Creates helper functions": any("get_partition_for_collection" in s for s in mock_op.conn.statements),
        }

        all_passed = True
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}")
            if not passed:
                all_passed = False

        print("\n" + "=" * 80)
        if all_passed:
            print("SUCCESS: All critical operations are present in the migration!")
        else:
            print("WARNING: Some expected operations are missing!")

        return all_passed

    finally:
        # Restore original functions
        real_op.get_bind = original_get_bind


if __name__ == "__main__":
    success = extract_sql_statements()
    sys.exit(0 if success else 1)
