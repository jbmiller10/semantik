#!/usr/bin/env python3
"""
Verify the 100-partition migration has all required components.
"""

import re


def verify_migration_file():
    """Read and verify the migration file content."""

    migration_file = "/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py"

    with open(migration_file) as f:
        content = f.read()

    print("=" * 80)
    print("MIGRATION FILE VERIFICATION")
    print("=" * 80)

    # Check for critical components
    checks = {
        "Drop old chunks table": "DROP TABLE IF EXISTS chunks CASCADE" in content,
        "Drop partition_mappings": "DROP TABLE IF EXISTS partition_mappings CASCADE" in content,
        "Create chunks with LIST partitioning": "PARTITION BY LIST (mod(hashtext(collection_id::text), 100))" in content,
        "Create 100 partitions": "FOR i IN 0..99 LOOP" in content,
        "Partition naming (chunks_part_XX)": "chunks_part_%s" in content,
        "Create indexes on partitions": "CREATE INDEX idx_chunks_part_" in content,
        "Create partition_health view": "CREATE OR REPLACE VIEW partition_health" in content,
        "Create partition_distribution view": "CREATE OR REPLACE VIEW partition_distribution" in content,
        "Create get_partition_for_collection function": "CREATE OR REPLACE FUNCTION get_partition_for_collection" in content,
        "Create analyze_partition_skew function": "CREATE OR REPLACE FUNCTION analyze_partition_skew" in content,
        "Create collection_chunking_stats view": "CREATE MATERIALIZED VIEW collection_chunking_stats" in content,
        "Uses BIGSERIAL for id": "id BIGSERIAL" in content,
        "Uses UUID for collection_id": "collection_id UUID NOT NULL" in content,
        "Uses vector(1536) for embeddings": "embedding vector(1536)" in content,
        "Uses JSONB for metadata": "metadata JSONB DEFAULT '{}'" in content,
        "Primary key on (id, collection_id)": "PRIMARY KEY (id, collection_id)" in content,
        "Foreign key to collections": "FOREIGN KEY (collection_id) REFERENCES collections(id)" in content,
        "Has downgrade function": "def downgrade()" in content,
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    # Count SQL statements
    print("\n" + "=" * 80)
    print("STATEMENT COUNTS")
    print("=" * 80)

    drop_count = len(re.findall(r'DROP\s+(?:TABLE|VIEW|FUNCTION|MATERIALIZED VIEW)', content, re.IGNORECASE))
    create_table_count = len(re.findall(r'CREATE\s+TABLE', content, re.IGNORECASE))
    create_index_count = len(re.findall(r'CREATE\s+INDEX', content, re.IGNORECASE))
    create_view_count = len(re.findall(r'CREATE\s+(?:OR\s+REPLACE\s+)?VIEW', content, re.IGNORECASE))
    create_function_count = len(re.findall(r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION', content, re.IGNORECASE))

    print(f"DROP statements: {drop_count}")
    print(f"CREATE TABLE statements: {create_table_count}")
    print(f"CREATE INDEX statements: {create_index_count}")
    print(f"CREATE VIEW statements: {create_view_count}")
    print(f"CREATE FUNCTION statements: {create_function_count}")

    # Check partition loop details
    partition_loop = re.search(r'FOR i IN (\d+)\.\.(\d+) LOOP', content)
    if partition_loop:
        start, end = partition_loop.groups()
        partition_count = int(end) - int(start) + 1
        print(f"\nPartition loop: {start} to {end} = {partition_count} partitions")
        if partition_count != 100:
            print(f"⚠️  WARNING: Expected 100 partitions, found {partition_count}")
            all_passed = False
    else:
        print("\n⚠️  WARNING: Could not find partition loop")
        all_passed = False

    # Check for proper partition naming
    print("\n" + "=" * 80)
    print("PARTITION NAMING CHECK")
    print("=" * 80)

    if "LPAD(i::text, 2, '0')" in content:
        print("✓ Uses zero-padding for partition numbers (00-99)")
    else:
        print("✗ Missing zero-padding for partition numbers")
        all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_passed:
        print("✅ SUCCESS: Migration file has all required components!")
        print("\nKey features implemented:")
        print("- 100 direct LIST partitions using mod(hashtext(collection_id), 100)")
        print("- Monitoring views for partition health and distribution")
        print("- Helper functions for partition management")
        print("- Proper indexes on each partition")
        print("- Clean downgrade path")
    else:
        print("❌ FAILURE: Some required components are missing!")

    return all_passed


if __name__ == "__main__":
    import sys
    success = verify_migration_file()
    sys.exit(0 if success else 1)
