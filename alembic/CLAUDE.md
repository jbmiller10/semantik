<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding new migrations
     - Changing partitioning strategy
     - Modifying migration patterns
     - Altering table schemas
     Keep this documentation in sync with the actual implementation! -->

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
    from alembic.migrations_utils.migration_safety import (
        require_destructive_flag,
        safe_drop_table,
        create_table_backup,
        verify_backup
    )
    
    def upgrade():
        conn = op.get_bind()
        
        # Require explicit permission for destructive operations
        require_destructive_flag("DROP TABLE chunks CASCADE")
        
        # Create backup before DROP
        backup_table_name, row_count = create_table_backup(
            conn, "chunks", revision, check_exists=True
        )
        
        # Verify backup integrity
        if backup_table_name and row_count > 0:
            if not verify_backup(conn, backup_table_name, row_count):
                raise RuntimeError("Backup verification failed!")
        
        # Safe drop with automatic backup
        safe_drop_table(conn, "chunks", revision, cascade=True, backup=True)
  </safe-migration>
  
  <safety-requirements>
    <!-- Environment variable required for destructive operations -->
    ALLOW_DESTRUCTIVE_MIGRATIONS=true
    
    <!-- Backup retention (default 7 days) -->
    MIGRATION_BACKUP_RETENTION_DAYS=7
  </safety-requirements>
  
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

<backup-utilities>
  <module>alembic/migrations_utils/migration_safety.py</module>
  <functions>
    <function>require_destructive_flag() - Checks ALLOW_DESTRUCTIVE_MIGRATIONS env var</function>
    <function>create_table_backup() - Creates timestamped backup before DROP</function>
    <function>verify_backup() - Verifies backup integrity</function>
    <function>safe_drop_table() - Wraps DROP TABLE with safety checks</function>
    <function>restore_from_backup() - Restores table from backup</function>
  </functions>
  
  <backup-manager>
    <!-- Command-line backup management -->
    poetry run python alembic/migrations_utils/backup_manager.py \
      --database-url $DATABASE_URL \
      --action list  # list, cleanup, verify, extend, status
  </backup-manager>
</backup-utilities>

<common-issues>
  <issue>
    <problem>Migration fails with async driver error</problem>
    <solution>env.py automatically converts asyncpg to psycopg2</solution>
  </issue>
  <issue>
    <problem>Partition key not calculated</problem>
    <solution>Check chunk_partition_trigger exists and is enabled</solution>
  </issue>
  <issue>
    <problem>Destructive migration blocked</problem>
    <solution>Set ALLOW_DESTRUCTIVE_MIGRATIONS=true to proceed</solution>
  </issue>
  <issue>
    <problem>Backup verification fails</problem>
    <solution>Check migration_backups table and disk space</solution>
  </issue>
</common-issues>