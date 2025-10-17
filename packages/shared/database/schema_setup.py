"""Helpers to ensure the chunking schema matches production expectations.

These helpers are intentionally idempotent so they can be invoked from both
Alembic migrations and test setup code. They create the partitioned ``chunks``
table, trigger functions, helper functions, monitoring views, and supporting
indexes that production relies on.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

PARTITION_COUNT = 100


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _scalar(conn: Connection, sql: str, **params: Any) -> Any:
    """Execute a scalar query."""
    return conn.execute(text(sql), params).scalar()


def _table_exists(conn: Connection, table_name: str) -> bool:
    """Return True if the table exists in the public schema."""
    return bool(
        _scalar(
            conn,
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = :table_name
            )
            """,
            table_name=table_name,
        )
    )


def _column_exists(conn: Connection, table_name: str, column_name: str) -> bool:
    """Return True if the column exists on the given table."""
    return bool(
        _scalar(
            conn,
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = :table_name
                  AND column_name = :column_name
            )
            """,
            table_name=table_name,
            column_name=column_name,
        )
    )


def _column_data_type(conn: Connection, table_name: str, column_name: str) -> str | None:
    """Return the PostgreSQL data type for the column."""
    return _scalar(
        conn,
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = :table_name
          AND column_name = :column_name
        """,
        table_name=table_name,
        column_name=column_name,
    )


def _index_exists(conn: Connection, index_name: str) -> bool:
    """Return True if the index exists."""
    return bool(
        _scalar(
            conn,
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND indexname = :index_name
            )
            """,
            index_name=index_name,
        )
    )


def _function_exists(conn: Connection, function_name: str) -> bool:
    """Return True if a function with the given name exists in the public schema."""
    return bool(
        _scalar(
            conn,
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_proc
                JOIN pg_namespace ns ON ns.oid = pg_proc.pronamespace
                WHERE ns.nspname = 'public'
                  AND pg_proc.proname = :function_name
            )
            """,
            function_name=function_name,
        )
    )


def _view_exists(conn: Connection, view_name: str, *, materialized: bool = False) -> bool:
    """Return True if the view (or materialized view) exists."""
    if materialized:
        sql = """
            SELECT EXISTS (
                SELECT 1
                FROM pg_matviews
                WHERE schemaname = 'public'
                  AND matviewname = :view_name
            )
        """
    else:
        sql = """
            SELECT EXISTS (
                SELECT 1
                FROM pg_views
                WHERE schemaname = 'public'
                  AND viewname = :view_name
            )
        """
    return bool(_scalar(conn, sql, view_name=view_name))


def _is_partitioned_table(conn: Connection, table_name: str) -> bool:
    """Return True if the table is partitioned."""
    return bool(
        _scalar(
            conn,
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_partitioned_table p
                JOIN pg_class c ON p.partrelid = c.oid
                WHERE c.relname = :table_name
            )
            """,
            table_name=table_name,
        )
    )


# ---------------------------------------------------------------------------
# Core chunk table and partitions
# ---------------------------------------------------------------------------

def _create_partitioned_chunks_table(conn: Connection) -> None:
    """Create the partitioned chunks table from scratch."""
    conn.execute(
        text(
            f"""
            CREATE TABLE chunks (
                id BIGSERIAL,
                collection_id VARCHAR NOT NULL,
                partition_key INTEGER NOT NULL,
                document_id VARCHAR,
                chunking_config_id INTEGER,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_offset INTEGER,
                end_offset INTEGER,
                token_count INTEGER,
                embedding_vector_id VARCHAR,
                metadata JSONB DEFAULT '{{}}'::JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (id, collection_id, partition_key),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
            ) PARTITION BY LIST (partition_key)
            """
        )
    )


def _recreate_as_partitioned(conn: Connection) -> None:
    """Promote an existing non-partitioned chunks table to a partitioned version."""
    conn.execute(text("ALTER TABLE chunks RENAME TO chunks_legacy"))
    conn.execute(text("ALTER SEQUENCE IF EXISTS chunks_id_seq RENAME TO chunks_legacy_id_seq"))

    _create_partitioned_chunks_table(conn)

    # Create partitions before copying to avoid routing failures
    _ensure_chunk_partitions(conn)

    # Copy data from legacy table if any rows exist
    conn.execute(
        text(
            """
            INSERT INTO chunks (
                id,
                collection_id,
                document_id,
                chunking_config_id,
                chunk_index,
                content,
                start_offset,
                end_offset,
                token_count,
                embedding_vector_id,
                metadata,
                created_at,
                updated_at
            )
            SELECT
                id,
                collection_id,
                document_id,
                chunking_config_id,
                chunk_index,
                content,
                start_offset,
                end_offset,
                token_count,
                embedding_vector_id,
                metadata,
                created_at,
                updated_at
            FROM chunks_legacy
            """
        )
    )

    conn.execute(text("DROP TABLE chunks_legacy CASCADE"))

    # Reset the sequence to match the new max id
    conn.execute(
        text(
            """
            SELECT setval(
                'chunks_id_seq',
                COALESCE((SELECT MAX(id) FROM chunks), 0) + 1,
                FALSE
            )
            """
        )
    )


def _ensure_chunk_table(conn: Connection) -> None:
    """Ensure the chunks table exists and is partitioned."""
    table_exists = _table_exists(conn, "chunks")
    if not table_exists:
        _create_partitioned_chunks_table(conn)
        return

    if not _is_partitioned_table(conn, "chunks"):
        _recreate_as_partitioned(conn)
        return

    strategy = _scalar(
        conn,
        """
        SELECT partstrat
        FROM pg_partitioned_table p
        JOIN pg_class c ON p.partrelid = c.oid
        WHERE c.relname = :table_name
        """,
        table_name="chunks",
    )
    if strategy != "l":
        _recreate_as_partitioned(conn)


def _ensure_chunk_partitions(conn: Connection) -> None:
    """Create any missing LIST partitions and required indexes."""
    for i in range(PARTITION_COUNT):
        partition_name = f"chunks_part_{i:02d}"
        if not _table_exists(conn, partition_name):
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {partition_name}
                    PARTITION OF chunks
                    FOR VALUES IN ({i})
                    """
                )
            )

        # Per-partition indexes
        indexes = {
            f"idx_{partition_name}_collection": f"CREATE INDEX {{name}} ON {partition_name}(collection_id)",
            f"idx_{partition_name}_document": f"CREATE INDEX {{name}} ON {partition_name}(document_id) "
            "WHERE document_id IS NOT NULL",
            f"idx_{partition_name}_created": f"CREATE INDEX {{name}} ON {partition_name}(created_at)",
            f"idx_{partition_name}_chunk_index": f"CREATE INDEX {{name}} ON {partition_name}(collection_id, chunk_index)",
            f"idx_{partition_name}_collection_document": f"CREATE INDEX {{name}} "
            f"ON {partition_name}(collection_id, document_id)",
            f"idx_{partition_name}_document_index": f"CREATE INDEX {{name}} "
            f"ON {partition_name}(document_id, chunk_index)",
            f"idx_{partition_name}_created_brin": f"CREATE INDEX {{name}} "
            f"ON {partition_name} USING brin(created_at)",
            f"idx_{partition_name}_no_embedding": f"CREATE INDEX {{name}} "
            f"ON {partition_name}(collection_id, id) WHERE embedding_vector_id IS NULL",
        }

        for index_name, ddl_template in indexes.items():
            if not _index_exists(conn, index_name):
                conn.execute(text(ddl_template.format(name=index_name)))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ensure_chunk_functions(conn: Connection) -> None:
    """Create helper functions used by repositories."""
    conn.execute(
        text(
            f"""
            CREATE OR REPLACE FUNCTION get_partition_key(collection_id VARCHAR)
            RETURNS INTEGER AS $$
            BEGIN
                RETURN abs(hashtext(collection_id::text)) % {PARTITION_COUNT};
            END;
            $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
            """
        )
    )

    conn.execute(
        text(
            f"""
            CREATE OR REPLACE FUNCTION get_partition_for_collection(collection_id VARCHAR)
            RETURNS TEXT AS $$
            DECLARE
                key INTEGER;
            BEGIN
                key := abs(hashtext(collection_id::text)) % {PARTITION_COUNT};
                RETURN 'chunks_part_' || LPAD(key::text, 2, '0');
            END;
            $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
            """
        )
    )

    conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION refresh_collection_chunking_stats()
            RETURNS void AS $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_matviews WHERE schemaname = 'public' AND matviewname = 'collection_chunking_stats'
                ) THEN
                    BEGIN
                        REFRESH MATERIALIZED VIEW CONCURRENTLY collection_chunking_stats;
                    EXCEPTION WHEN feature_not_supported THEN
                        -- FALLBACK when CONCURRENTLY isn't available (e.g., inside a transaction)
                        REFRESH MATERIALIZED VIEW collection_chunking_stats;
                    END;
                END IF;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )


# ---------------------------------------------------------------------------
# Monitoring views and functions
# ---------------------------------------------------------------------------

def _ensure_partition_views(conn: Connection) -> None:
    """Create monitoring views relied on by services and tests."""

    # Drop existing views to avoid column-mismatch errors when replacing definitions
    views_to_reset = [
        "partition_health_summary",
        "partition_hot_spots",
        "partition_distribution",
        "partition_health",
        "partition_chunk_distribution",
        "partition_size_distribution",
        "active_chunking_configs",
    ]
    for view_name in views_to_reset:
        conn.execute(text(f"DROP VIEW IF EXISTS {view_name} CASCADE"))

    # Base size distribution view
    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_size_distribution AS
            SELECT
                CAST(SUBSTRING(c.relname FROM 'chunks_part_([0-9]+)') AS INTEGER) AS partition_num,
                c.relname AS partition_name,
                COALESCE(pg_total_relation_size(c.oid), 0) AS size_bytes,
                pg_size_pretty(COALESCE(pg_total_relation_size(c.oid), 0)) AS partition_size,
                COALESCE(st.n_live_tup, 0) AS estimated_rows,
                COALESCE(st.n_dead_tup, 0) AS dead_rows,
                st.last_vacuum,
                st.last_autovacuum
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_stat_all_tables st ON st.relid = c.oid
            WHERE c.relname LIKE 'chunks_part_%'
              AND n.nspname = 'public';
            """
        )
    )

    # Chunk distribution view
    conn.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW partition_chunk_distribution AS
            WITH agg AS (
                SELECT
                    partition_key AS partition_num,
                    COUNT(*) AS chunk_count,
                    COUNT(DISTINCT collection_id) AS collection_count,
                    COUNT(DISTINCT document_id) AS document_count,
                    COALESCE(AVG(NULLIF(token_count, 0)), 0) AS avg_token_count,
                    COALESCE(SUM(token_count), 0) AS total_tokens,
                    MAX(created_at) AS newest_chunk,
                    MIN(created_at) AS oldest_chunk
                FROM chunks
                GROUP BY partition_key
            )
            SELECT
                gs.partition_num,
                COALESCE(agg.chunk_count, 0) AS chunk_count,
                COALESCE(agg.collection_count, 0) AS collection_count,
                COALESCE(agg.document_count, 0) AS document_count,
                COALESCE(agg.avg_token_count, 0) AS avg_token_count,
                COALESCE(agg.total_tokens, 0) AS total_tokens,
                agg.newest_chunk,
                agg.oldest_chunk
            FROM generate_series(0, {PARTITION_COUNT - 1}) AS gs(partition_num)
            LEFT JOIN agg ON agg.partition_num = gs.partition_num
            ORDER BY gs.partition_num;
            """
        )
    )

    # Hot partitions view
    conn.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW partition_hot_spots AS
            WITH counts AS (
                SELECT
                    partition_key AS partition_num,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 hour') AS chunks_last_hour,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day') AS chunks_last_day,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 week') AS chunks_last_week
                FROM chunks
                GROUP BY partition_key
            ),
            totals AS (
                SELECT
                    SUM(chunks_last_hour) AS total_last_hour,
                    SUM(chunks_last_day) AS total_last_day,
                    SUM(chunks_last_week) AS total_last_week
                FROM counts
            )
            SELECT
                gs.partition_num,
                COALESCE(c.chunks_last_hour, 0) AS chunks_last_hour,
                COALESCE(c.chunks_last_day, 0) AS chunks_last_day,
                COALESCE(c.chunks_last_week, 0) AS chunks_last_week,
                CASE
                    WHEN totals.total_last_hour > 0
                        THEN ROUND((COALESCE(c.chunks_last_hour, 0)::NUMERIC / totals.total_last_hour) * 100, 2)
                    ELSE 0
                END AS hour_percentage,
                CASE
                    WHEN totals.total_last_day > 0
                        THEN ROUND((COALESCE(c.chunks_last_day, 0)::NUMERIC / totals.total_last_day) * 100, 2)
                    ELSE 0
                END AS day_percentage,
                CASE
                    WHEN totals.total_last_week > 0
                        THEN ROUND((COALESCE(c.chunks_last_week, 0)::NUMERIC / totals.total_last_week) * 100, 2)
                    ELSE 0
                END AS week_percentage
            FROM generate_series(0, {PARTITION_COUNT - 1}) AS gs(partition_num)
            LEFT JOIN counts c ON c.partition_num = gs.partition_num
            CROSS JOIN totals
            ORDER BY gs.partition_num;
            """
        )
    )

    # Partition health view (per-partition detail)
    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_health AS
            WITH stats AS (
                SELECT
                    psd.partition_num,
                    psd.partition_name,
                    pcd.chunk_count,
                    pcd.collection_count,
                    pcd.document_count,
                    pcd.total_tokens,
                    psd.size_bytes,
                    psd.partition_size,
                    psd.estimated_rows,
                    psd.dead_rows,
                    psd.last_vacuum,
                    psd.last_autovacuum
                FROM partition_size_distribution psd
                LEFT JOIN partition_chunk_distribution pcd ON psd.partition_num = pcd.partition_num
            ),
            summary AS (
                SELECT
                    COALESCE(SUM(chunk_count), 0) AS total_chunks,
                    COALESCE(AVG(NULLIF(chunk_count, 0)), 0) AS avg_chunks,
                    COALESCE(STDDEV_POP(chunk_count), 0) AS stddev_chunks
                FROM partition_chunk_distribution
            )
            SELECT
                stats.partition_num,
                stats.partition_name,
                COALESCE(stats.chunk_count, 0) AS chunk_count,
                COALESCE(stats.collection_count, 0) AS collection_count,
                COALESCE(stats.document_count, 0) AS document_count,
                COALESCE(stats.total_tokens, 0) AS total_tokens,
                COALESCE(stats.size_bytes, 0) AS size_bytes,
                stats.partition_size,
                stats.estimated_rows,
                stats.dead_rows,
                stats.last_vacuum,
                stats.last_autovacuum,
                CASE
                    WHEN summary.avg_chunks = 0 THEN 0
                    ELSE ROUND(((COALESCE(stats.chunk_count, 0) - summary.avg_chunks) / summary.avg_chunks) * 100, 2)
                END AS pct_deviation_from_avg,
                CASE
                    WHEN summary.avg_chunks = 0 THEN 'NORMAL'
                    WHEN COALESCE(stats.chunk_count, 0) > summary.avg_chunks * 1.2 THEN 'HOT'
                    WHEN COALESCE(stats.chunk_count, 0) < summary.avg_chunks * 0.8 THEN 'COLD'
                    ELSE 'NORMAL'
                END AS partition_status,
                CASE
                    WHEN stats.dead_rows > GREATEST(stats.estimated_rows * 0.1, 1000) THEN TRUE
                    ELSE FALSE
                END AS needs_vacuum
            FROM stats
            CROSS JOIN summary
            ORDER BY stats.partition_num;
            """
        )
    )

    # Partition distribution summary view
    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_distribution AS
            WITH partition_counts AS (
                SELECT
                    partition_num,
                    chunk_count
                FROM partition_chunk_distribution
            ),
            summary AS (
                SELECT
                    COUNT(*) FILTER (WHERE chunk_count > 0) AS partitions_used,
                    COUNT(*) FILTER (WHERE chunk_count = 0) AS empty_partitions,
                    COALESCE(AVG(NULLIF(chunk_count, 0)), 0) AS avg_chunks_per_partition,
                    COALESCE(STDDEV_POP(chunk_count), 0) AS stddev_chunks,
                    COALESCE(MAX(chunk_count), 0) AS max_chunks,
                    COALESCE(MIN(chunk_count), 0) AS min_chunks
                FROM partition_counts
            )
            SELECT
                summary.partitions_used,
                summary.empty_partitions,
                summary.avg_chunks_per_partition,
                summary.stddev_chunks,
                CASE
                    WHEN summary.avg_chunks_per_partition = 0 THEN 0
                    ELSE ROUND(summary.stddev_chunks / summary.avg_chunks_per_partition, 4)
                END AS max_skew_ratio,
                summary.max_chunks,
                summary.min_chunks,
                CASE
                    WHEN summary.avg_chunks_per_partition = 0 THEN 'HEALTHY'
                    WHEN summary.stddev_chunks / NULLIF(summary.avg_chunks_per_partition, 0) > 0.5 THEN 'REBALANCE NEEDED'
                    WHEN summary.stddev_chunks / NULLIF(summary.avg_chunks_per_partition, 0) > 0.3 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END AS distribution_status
            FROM summary;
            """
        )
    )

    # Partition health summary view
    conn.execute(
        text(
            f"""
            CREATE OR REPLACE VIEW partition_health_summary AS
            WITH chunk_totals AS (
                SELECT COALESCE(SUM(chunk_count), 0) AS total_chunks
                FROM partition_chunk_distribution
            ),
            size_totals AS (
                SELECT COALESCE(SUM(size_bytes), 0) AS total_size
                FROM partition_size_distribution
            )
            SELECT
                pcd.partition_num,
                pcd.chunk_count,
                chunk_totals.total_chunks,
                CASE
                    WHEN chunk_totals.total_chunks = 0 THEN 0
                    ELSE ROUND((pcd.chunk_count::NUMERIC / chunk_totals.total_chunks) * 100, 2)
                END AS chunk_percentage,
                CASE
                    WHEN size_totals.total_size = 0 THEN 0
                    ELSE ROUND((COALESCE(psd.size_bytes, 0)::NUMERIC / size_totals.total_size) * 100, 2)
                END AS size_percentage,
                CASE
                    WHEN chunk_totals.total_chunks = 0 THEN 0
                    ELSE ROUND((pcd.chunk_count::NUMERIC / chunk_totals.total_chunks) / (1.0 / {PARTITION_COUNT}), 4)
                END AS chunk_skew,
                CASE
                    WHEN size_totals.total_size = 0 THEN 0
                    ELSE ROUND((COALESCE(psd.size_bytes, 0)::NUMERIC / size_totals.total_size) / (1.0 / {PARTITION_COUNT}), 4)
                END AS size_skew,
                CASE
                    WHEN chunk_totals.total_chunks = 0 THEN 'HEALTHY'
                    WHEN (pcd.chunk_count::NUMERIC / chunk_totals.total_chunks) * 100 > 16.25 THEN 'UNBALANCED'
                    WHEN (pcd.chunk_count::NUMERIC / chunk_totals.total_chunks) * 100 > 11.25 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END AS health_status,
                CASE
                    WHEN (pcd.chunk_count::NUMERIC / NULLIF(chunk_totals.total_chunks, 0)) * 100 > 16.25
                        THEN 'Immediate rebalancing recommended'
                    WHEN (pcd.chunk_count::NUMERIC / NULLIF(chunk_totals.total_chunks, 0)) * 100 > 11.25
                        THEN 'Monitor and plan rebalancing'
                    ELSE NULL
                END AS recommendation
            FROM partition_chunk_distribution pcd
            LEFT JOIN partition_size_distribution psd ON psd.partition_num = pcd.partition_num
            CROSS JOIN chunk_totals
            CROSS JOIN size_totals
            ORDER BY pcd.partition_num;
            """
        )
    )

    # Analyze partition skew function
    conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION analyze_partition_skew()
            RETURNS TABLE (
                metric TEXT,
                value NUMERIC,
                status TEXT,
                details TEXT
            ) AS $$
            DECLARE
                max_chunk_skew NUMERIC := 0;
                avg_chunk_skew NUMERIC := 0;
                max_size_skew NUMERIC := 0;
                warning_partitions INTEGER := 0;
                unbalanced_partitions INTEGER := 0;
            BEGIN
                SELECT
                    COALESCE(MAX(chunk_skew), 0),
                    COALESCE(AVG(chunk_skew), 0),
                    COALESCE(MAX(size_skew), 0),
                    COUNT(*) FILTER (WHERE health_status = 'WARNING'),
                    COUNT(*) FILTER (WHERE health_status = 'UNBALANCED')
                INTO
                    max_chunk_skew,
                    avg_chunk_skew,
                    max_size_skew,
                    warning_partitions,
                    unbalanced_partitions
                FROM partition_health_summary;

                RETURN QUERY
                SELECT
                    'max_chunk_skew',
                    max_chunk_skew,
                    CASE
                        WHEN max_chunk_skew > 1.5 THEN 'CRITICAL'
                        WHEN max_chunk_skew > 1.2 THEN 'WARNING'
                        ELSE 'HEALTHY'
                    END,
                    'Highest chunk skew ratio across partitions';

                RETURN QUERY
                SELECT
                    'average_chunk_skew',
                    avg_chunk_skew,
                    CASE
                        WHEN avg_chunk_skew > 1.3 THEN 'WARNING'
                        ELSE 'HEALTHY'
                    END,
                    'Average skew ratio (1.0 == perfectly balanced)';

                RETURN QUERY
                SELECT
                    'max_size_skew',
                    max_size_skew,
                    CASE
                        WHEN max_size_skew > 1.5 THEN 'WARNING'
                        ELSE 'HEALTHY'
                    END,
                    'Largest storage skew ratio across partitions';

                RETURN QUERY
                SELECT
                    'warning_partitions',
                    warning_partitions,
                    CASE
                        WHEN warning_partitions > 0 THEN 'WARNING'
                        ELSE 'HEALTHY'
                    END,
                    'Partitions currently flagged with WARNING status';

                RETURN QUERY
                SELECT
                    'unbalanced_partitions',
                    unbalanced_partitions,
                    CASE
                        WHEN unbalanced_partitions > 0 THEN 'CRITICAL'
                        ELSE 'HEALTHY'
                    END,
                    'Partitions currently flagged with UNBALANCED status';

                RETURN;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )

    # Active chunking configs view
    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW active_chunking_configs AS
            SELECT
                cc.id,
                cc.strategy_id,
                cs.name AS strategy_name,
                cc.config_hash,
                cc.config_data,
                cc.created_at,
                cc.use_count,
                cc.last_used_at,
                sub.collections_using
            FROM chunking_configs cc
            JOIN chunking_strategies cs ON cs.id = cc.strategy_id
            LEFT JOIN (
                SELECT
                    chunking_config_id,
                    COUNT(DISTINCT collection_id) AS collections_using
                FROM chunks
                WHERE chunking_config_id IS NOT NULL
                GROUP BY chunking_config_id
            ) sub ON sub.chunking_config_id = cc.id
            WHERE cc.use_count > 0;
            """
        )
    )


# ---------------------------------------------------------------------------
# Materialized view and indexes
# ---------------------------------------------------------------------------

def _ensure_collection_chunking_stats(conn: Connection) -> None:
    """Create materialized view used by monitoring and ensure indexes."""
    if not _view_exists(conn, "collection_chunking_stats", materialized=True):
        conn.execute(
            text(
                """
                CREATE MATERIALIZED VIEW collection_chunking_stats AS
                SELECT
                    c.id AS collection_id,
                    c.name,
                    COUNT(ch.id) AS total_chunks,
                    COUNT(DISTINCT ch.document_id) AS chunked_documents,
                    COALESCE(SUM(ch.token_count), 0) AS total_tokens,
                    COALESCE(AVG(NULLIF(ch.token_count, 0)), 0) AS avg_tokens_per_chunk,
                    MAX(ch.created_at) AS last_chunk_created
                FROM collections c
                LEFT JOIN chunks ch ON ch.collection_id = c.id
                GROUP BY c.id, c.name
                WITH DATA;
                """
            )
        )

    if not _index_exists(conn, "ix_collection_chunking_stats_id"):
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX ix_collection_chunking_stats_id
                ON collection_chunking_stats (collection_id)
                """
            )
        )


# ---------------------------------------------------------------------------
# Table column and index hygiene
# ---------------------------------------------------------------------------

def _ensure_collection_columns(conn: Connection) -> None:
    """Add chunking-related columns to collections/documents if missing."""
    if not _column_exists(conn, "collections", "default_chunking_config_id"):
        conn.execute(
            text(
                """
                ALTER TABLE collections
                ADD COLUMN default_chunking_config_id INTEGER NULL
                REFERENCES chunking_configs(id)
                """
            )
        )
    if not _column_exists(conn, "collections", "chunks_total_count"):
        conn.execute(text("ALTER TABLE collections ADD COLUMN chunks_total_count INTEGER NOT NULL DEFAULT 0"))
    if not _column_exists(conn, "collections", "chunking_completed_at"):
        conn.execute(text("ALTER TABLE collections ADD COLUMN chunking_completed_at TIMESTAMPTZ NULL"))
    if not _column_exists(conn, "collections", "chunking_strategy"):
        conn.execute(text("ALTER TABLE collections ADD COLUMN chunking_strategy VARCHAR NULL"))
    if not _column_exists(conn, "collections", "chunking_config"):
        conn.execute(text("ALTER TABLE collections ADD COLUMN chunking_config JSONB NULL"))

    # Ensure total_size_bytes uses BIGINT to avoid overflow during tests
    total_size_type = _column_data_type(conn, "collections", "total_size_bytes")
    if total_size_type and total_size_type.lower() != "bigint":
        conn.execute(
            text(
                """
                ALTER TABLE collections
                ALTER COLUMN total_size_bytes TYPE BIGINT
                USING total_size_bytes::BIGINT
                """
            )
        )

    if not _index_exists(conn, "ix_collections_default_chunking_config_id"):
        conn.execute(
            text(
                """
                CREATE INDEX ix_collections_default_chunking_config_id
                ON collections(default_chunking_config_id)
                """
            )
        )
    if not _index_exists(conn, "idx_collections_owner_status"):
        conn.execute(
            text(
                """
                CREATE INDEX idx_collections_owner_status
                ON collections(owner_id, status)
                """
            )
        )

    if not _column_exists(conn, "documents", "chunking_config_id"):
        conn.execute(
            text(
                """
                ALTER TABLE documents
                ADD COLUMN chunking_config_id INTEGER NULL
                REFERENCES chunking_configs(id)
                """
            )
        )
    if not _column_exists(conn, "documents", "chunks_count"):
        conn.execute(text("ALTER TABLE documents ADD COLUMN chunks_count INTEGER NOT NULL DEFAULT 0"))
    if not _column_exists(conn, "documents", "chunking_started_at"):
        conn.execute(text("ALTER TABLE documents ADD COLUMN chunking_started_at TIMESTAMPTZ NULL"))
    if not _column_exists(conn, "documents", "chunking_completed_at"):
        conn.execute(text("ALTER TABLE documents ADD COLUMN chunking_completed_at TIMESTAMPTZ NULL"))

    if not _index_exists(conn, "ix_documents_chunking_config_id"):
        conn.execute(
            text(
                """
                CREATE INDEX ix_documents_chunking_config_id
                ON documents(chunking_config_id)
                """
            )
        )
    if not _index_exists(conn, "ix_documents_collection_id_chunking_completed_at"):
        conn.execute(
            text(
                """
                CREATE INDEX ix_documents_collection_id_chunking_completed_at
                ON documents(collection_id, chunking_completed_at)
                """
            )
        )
    if not _index_exists(conn, "idx_documents_collection_status"):
        conn.execute(
            text(
                """
                CREATE INDEX idx_documents_collection_status
                ON documents(collection_id, status)
                """
            )
        )

    if not _index_exists(conn, "idx_operations_collection_type_status"):
        conn.execute(
            text(
                """
                CREATE INDEX idx_operations_collection_type_status
                ON operations(collection_id, type, status)
                """
            )
        )
    if not _index_exists(conn, "idx_operations_created_desc"):
        conn.execute(
            text(
                """
                CREATE INDEX idx_operations_created_desc
                ON operations((created_at DESC))
                """
            )
        )
    if not _index_exists(conn, "idx_operations_user_status"):
        conn.execute(
            text(
                """
                CREATE INDEX idx_operations_user_status
                ON operations(user_id, status)
                WHERE status IN ('processing', 'pending')
                """
            )
        )
    if not _index_exists(conn, "idx_operations_config_strategy"):
        conn.execute(
            text(
                """
                CREATE INDEX idx_operations_config_strategy
                ON operations((config->>'strategy'))
                WHERE config IS NOT NULL
                """
            )
        )


# ---------------------------------------------------------------------------
# Default data
# ---------------------------------------------------------------------------

def _ensure_default_strategies(conn: Connection) -> None:
    """Insert default chunking strategies if the table is empty."""
    strategy_count = _scalar(conn, "SELECT COUNT(*) FROM chunking_strategies")
    if not strategy_count:
        conn.execute(
            text(
                """
                INSERT INTO chunking_strategies (name, description, is_active, meta) VALUES
                ('character', 'Fixed-size character based chunking', true, '{"supports_streaming": true}'),
                ('recursive', 'Sentence-aware recursive chunking', true, '{"supports_streaming": true, "recommended_default": true}'),
                ('markdown', 'Markdown structure aware chunking', true, '{"supports_streaming": true, "file_types": [".md", ".mdx"]}'),
                ('semantic', 'Embedding-driven semantic splitter', false, '{"supports_streaming": false, "requires_embeddings": true}'),
                ('hierarchical', 'Hierarchical parent-child chunking', false, '{"supports_streaming": false}'),
                ('hybrid', 'Adaptive strategy that picks best chunker automatically', false, '{"supports_streaming": false}')
                ON CONFLICT (name) DO NOTHING
                """
            )
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def ensure_chunking_infrastructure(conn: Connection) -> None:
    """Ensure all database objects required for chunking exist."""
    _ensure_chunk_table(conn)
    _ensure_chunk_partitions(conn)
    _ensure_chunk_functions(conn)
    _ensure_collection_columns(conn)
    _ensure_partition_views(conn)
    _ensure_collection_chunking_stats(conn)
    _ensure_default_strategies(conn)


def drop_chunking_infrastructure(conn: Connection) -> None:
    """Drop chunking-related objects. Primarily used for Alembic downgrades."""
    # Drop dependent views first
    views = [
        "active_chunking_configs",
        "partition_health_summary",
        "partition_hot_spots",
        "partition_distribution",
        "partition_health",
        "partition_chunk_distribution",
        "partition_size_distribution",
    ]
    for view in views:
        conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))

    conn.execute(text("DROP FUNCTION IF EXISTS analyze_partition_skew() CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS refresh_collection_chunking_stats() CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS get_partition_for_collection(VARCHAR) CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS get_partition_key(VARCHAR) CASCADE"))

    conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE"))

    conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks"))

    # Drop partitions and base table
    for i in range(PARTITION_COUNT):
        partition_name = f"chunks_part_{i:02d}"
        conn.execute(text(f"DROP TABLE IF EXISTS {partition_name} CASCADE"))

    conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))

    # Clean up indexes created on other tables
    indexes = [
        "ix_collections_default_chunking_config_id",
        "idx_collections_owner_status",
        "ix_documents_chunking_config_id",
        "ix_documents_collection_id_chunking_completed_at",
        "idx_documents_collection_status",
        "idx_operations_collection_type_status",
        "idx_operations_created_desc",
        "idx_operations_user_status",
        "idx_operations_config_strategy",
    ]
    for index in indexes:
        conn.execute(text(f"DROP INDEX IF EXISTS {index}"))

    # Drop added columns (safe if they exist)
    columns_sql = [
        ("collections", "default_chunking_config_id"),
        ("collections", "chunks_total_count"),
        ("collections", "chunking_completed_at"),
        ("collections", "chunking_strategy"),
        ("collections", "chunking_config"),
        ("documents", "chunking_config_id"),
        ("documents", "chunks_count"),
        ("documents", "chunking_started_at"),
        ("documents", "chunking_completed_at"),
    ]
    for table_name, column_name in columns_sql:
        if _column_exists(conn, table_name, column_name):
            conn.execute(text(f'ALTER TABLE {table_name} DROP COLUMN IF EXISTS "{column_name}" CASCADE'))
