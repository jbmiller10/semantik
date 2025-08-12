<component>
  <name>Database Migrations</name>
  <purpose>Alembic migrations for PostgreSQL schema management</purpose>
  <location>alembic/</location>
</component>

<critical-migrations>
  <migration file="005a8fe3aedc_initial_unified_schema_for_collections_.py">
    Initial schema with collections, documents, operations
  </migration>
  <migration file="ae558c9e183f_implement_100_direct_list_partitions.py">
    100-partition LIST partitioning for chunks table
  </migration>
  <migration file="52db15bd2686_add_chunking_tables_with_partitioning.py">
    Chunking strategy fields and configuration
  </migration>
</critical-migrations>

<partitioning-strategy>
  <table>chunks</table>
  <method>LIST partitioning on partition_key</method>
  <calculation>abs(hashtext(collection_id::text)) % 100</calculation>
  <partitions>100 partitions (chunk_part_0 to chunk_part_99)</partitions>
  <trigger>chunk_partition_trigger calculates partition_key</trigger>
</partitioning-strategy>

<migration-patterns>
  <safe-migration>
    <!-- Always backup before destructive changes -->
    from alembic.migrations_utils.backup_manager import BackupManager
    
    def upgrade():
        backup_manager = BackupManager()
        backup_manager.create_backup("pre_migration_backup")
        # ... migration code ...
  </safe-migration>
  
  <async-handling>
    <!-- Convert async driver for Alembic -->
    if connection.dialect.name == "postgresql" and "async" in str(connection.dialect):
        from sqlalchemy import create_engine
        sync_url = str(connection.engine.url).replace("+asyncpg", "")
        sync_engine = create_engine(sync_url)
  </async-handling>
</migration-patterns>

<running-migrations>
  <command>poetry run alembic upgrade head</command>
  <rollback>poetry run alembic downgrade -1</command>
  <status>poetry run alembic current</command>
  <history>poetry run alembic history</history>
</running-migrations>

<common-issues>
  <issue>
    <problem>Migration fails with async driver error</problem>
    <solution>env.py automatically converts asyncpg to psycopg2</solution>
  </issue>
  <issue>
    <problem>Partition key not calculated</problem>
    <solution>Check chunk_partition_trigger exists and is enabled</solution>
  </issue>
</common-issues>