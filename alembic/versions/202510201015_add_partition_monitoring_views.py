"""Add partition monitoring views and skew analysis helper.

Revision ID: 202510201015
Revises: f1a2b3c4d5e6
Create Date: 2025-10-20 10:15:00.000000

"""

from collections.abc import Sequence

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202510201015"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create partition monitoring views used by the API and maintenance tools."""

    conn = op.get_bind()

    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_chunk_distribution AS
            WITH partition_count AS (
                SELECT GREATEST(COUNT(*), 1) AS total_partitions
                FROM pg_inherits
                WHERE inhparent = 'chunks'::regclass
            ),
            series AS (
                SELECT generate_series(0, (SELECT total_partitions - 1 FROM partition_count)) AS partition_num
            ),
            chunk_counts AS (
                SELECT
                    partition_key AS partition_num,
                    COUNT(*)::bigint AS chunk_count,
                    COUNT(DISTINCT document_id)::bigint AS document_count,
                    COUNT(DISTINCT collection_id)::bigint AS collection_count,
                    COALESCE(AVG(COALESCE(token_count, 0)), 0)::numeric AS avg_tokens
                FROM chunks
                GROUP BY partition_key
            ),
            totals AS (
                SELECT COALESCE(SUM(chunk_count), 0)::bigint AS total_chunks
                FROM chunk_counts
            )
            SELECT
                s.partition_num,
                COALESCE(cc.chunk_count, 0)::bigint AS chunk_count,
                COALESCE(cc.document_count, 0)::bigint AS document_count,
                COALESCE(cc.collection_count, 0)::bigint AS collection_count,
                COALESCE(cc.avg_tokens, 0)::numeric AS avg_tokens,
                CASE
                    WHEN totals.total_chunks > 0
                        THEN ROUND(COALESCE(cc.chunk_count, 0)::numeric / totals.total_chunks * 100, 2)
                    ELSE 0::numeric
                END AS chunk_percentage
            FROM series s
            CROSS JOIN totals
            LEFT JOIN chunk_counts cc ON cc.partition_num = s.partition_num
            ORDER BY s.partition_num;
            """
        )
    )

    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_size_distribution AS
            WITH partition_count AS (
                SELECT GREATEST(COUNT(*), 1) AS total_partitions
                FROM pg_inherits
                WHERE inhparent = 'chunks'::regclass
            ),
            series AS (
                SELECT generate_series(0, (SELECT total_partitions - 1 FROM partition_count)) AS partition_num
            ),
            size_stats AS (
                SELECT
                    SUBSTRING(c.relname FROM 'chunks_part_([0-9]+)')::int AS partition_num,
                    pg_total_relation_size(c.oid) AS size_bytes,
                    COALESCE(c.reltuples, 0)::bigint AS row_estimate
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relname LIKE 'chunks_part_%'
                  AND c.relkind = 'r'
            ),
            totals AS (
                SELECT COALESCE(SUM(size_bytes), 0)::bigint AS total_size_bytes
                FROM size_stats
            )
            SELECT
                s.partition_num,
                COALESCE(ss.size_bytes, 0)::bigint AS size_bytes,
                pg_size_pretty(COALESCE(ss.size_bytes, 0)) AS size_pretty,
                CASE
                    WHEN totals.total_size_bytes > 0
                        THEN ROUND(COALESCE(ss.size_bytes, 0)::numeric / totals.total_size_bytes * 100, 2)
                    ELSE 0::numeric
                END AS size_percentage,
                COALESCE(ss.row_estimate, 0)::bigint AS row_estimate
            FROM series s
            CROSS JOIN totals
            LEFT JOIN size_stats ss ON ss.partition_num = s.partition_num
            ORDER BY s.partition_num;
            """
        )
    )

    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_hot_spots AS
            WITH partition_count AS (
                SELECT GREATEST(COUNT(*), 1) AS total_partitions
                FROM pg_inherits
                WHERE inhparent = 'chunks'::regclass
            ),
            series AS (
                SELECT generate_series(0, (SELECT total_partitions - 1 FROM partition_count)) AS partition_num
            ),
            activity AS (
                SELECT
                    partition_key AS partition_num,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 hour')::bigint AS chunks_last_hour,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '1 day')::bigint AS chunks_last_day,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 day')::bigint AS chunks_last_week
                FROM chunks
                GROUP BY partition_key
            ),
            totals AS (
                SELECT
                    COALESCE(SUM(chunks_last_hour), 0)::bigint AS total_hour,
                    COALESCE(SUM(chunks_last_day), 0)::bigint AS total_day,
                    COALESCE(SUM(chunks_last_week), 0)::bigint AS total_week
                FROM activity
            )
            SELECT
                s.partition_num,
                COALESCE(a.chunks_last_hour, 0)::bigint AS chunks_last_hour,
                COALESCE(a.chunks_last_day, 0)::bigint AS chunks_last_day,
                COALESCE(a.chunks_last_week, 0)::bigint AS chunks_last_week,
                CASE
                    WHEN totals.total_hour > 0
                        THEN ROUND(COALESCE(a.chunks_last_hour, 0)::numeric / totals.total_hour * 100, 2)
                    ELSE 0::numeric
                END AS hour_percentage,
                CASE
                    WHEN totals.total_day > 0
                        THEN ROUND(COALESCE(a.chunks_last_day, 0)::numeric / totals.total_day * 100, 2)
                    ELSE 0::numeric
                END AS day_percentage,
                CASE
                    WHEN totals.total_week > 0
                        THEN ROUND(COALESCE(a.chunks_last_week, 0)::numeric / totals.total_week * 100, 2)
                    ELSE 0::numeric
                END AS week_percentage
            FROM series s
            CROSS JOIN totals
            LEFT JOIN activity a ON a.partition_num = s.partition_num
            ORDER BY s.partition_num;
            """
        )
    )

    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_stats AS
            WITH latest_chunks AS (
                SELECT
                    partition_key AS partition_num,
                    MAX(created_at) AS last_created
                FROM chunks
                GROUP BY partition_key
            )
            SELECT
                dist.partition_num,
                dist.chunk_count,
                CASE
                    WHEN sizes.size_bytes > 0
                        THEN ROUND(sizes.size_bytes::numeric / 1048576, 4)
                    ELSE 0::numeric
                END AS total_size_mb,
                CASE
                    WHEN dist.chunk_count > 0 AND sizes.size_bytes > 0
                        THEN ROUND(sizes.size_bytes::numeric / dist.chunk_count / 1024, 4)
                    ELSE 0::numeric
                END AS avg_chunk_size_kb,
                latest.last_created AS created_at
            FROM partition_chunk_distribution dist
            LEFT JOIN partition_size_distribution sizes ON sizes.partition_num = dist.partition_num
            LEFT JOIN latest_chunks latest ON latest.partition_num = dist.partition_num
            ORDER BY dist.partition_num;
            """
        )
    )

    conn.execute(
        text(
            """
            CREATE OR REPLACE VIEW partition_health_summary AS
            WITH partition_count AS (
                SELECT GREATEST(COUNT(*), 1)::numeric AS total_partitions
                FROM pg_inherits
                WHERE inhparent = 'chunks'::regclass
            ),
            expected AS (
                SELECT 1.0 / total_partitions AS expected_fraction
                FROM partition_count
            ),
            chunk_data AS (
                SELECT
                    dist.partition_num,
                    dist.chunk_count,
                    dist.chunk_percentage,
                    SUM(dist.chunk_count) OVER ()::bigint AS total_chunks
                FROM partition_chunk_distribution dist
            ),
            size_data AS (
                SELECT
                    sizes.partition_num,
                    sizes.size_bytes,
                    sizes.size_percentage,
                    SUM(sizes.size_bytes) OVER ()::bigint AS total_size_bytes
                FROM partition_size_distribution sizes
            ),
            metrics AS (
                SELECT
                    cd.partition_num,
                    cd.chunk_count,
                    cd.total_chunks,
                    cd.chunk_percentage,
                    sd.size_percentage,
                    sd.size_bytes,
                    sd.total_size_bytes,
                    expected.expected_fraction,
                    CASE
                        WHEN expected.expected_fraction = 0 OR cd.total_chunks = 0 THEN 0
                        ELSE ABS((cd.chunk_count::numeric / cd.total_chunks) - expected.expected_fraction) / expected.expected_fraction
                    END AS chunk_skew,
                    CASE
                        WHEN expected.expected_fraction = 0 OR sd.total_size_bytes = 0 THEN 0
                        ELSE ABS((sd.size_bytes::numeric / sd.total_size_bytes) - expected.expected_fraction) / expected.expected_fraction
                    END AS size_skew
                FROM chunk_data cd
                JOIN size_data sd ON sd.partition_num = cd.partition_num
                CROSS JOIN expected
            )
            SELECT
                partition_num,
                chunk_count,
                total_chunks,
                ROUND(chunk_percentage, 2) AS chunk_percentage,
                ROUND(size_percentage, 2) AS size_percentage,
                CASE
                    WHEN GREATEST(chunk_skew, size_skew) >= 0.5 THEN 'UNBALANCED'
                    WHEN GREATEST(chunk_skew, size_skew) >= 0.3 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END AS health_status,
                ROUND(chunk_skew, 4) AS chunk_skew,
                ROUND(size_skew, 4) AS size_skew,
                CASE
                    WHEN GREATEST(chunk_skew, size_skew) >= 0.5 THEN 'Consider rebalancing affected partitions'
                    WHEN GREATEST(chunk_skew, size_skew) >= 0.3 THEN 'Monitor partition growth'
                    ELSE NULL
                END AS recommendation
            FROM metrics
            ORDER BY partition_num;
            """
        )
    )

    conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION analyze_partition_skew()
            RETURNS TABLE(metric text, value numeric, status text, details text)
            LANGUAGE plpgsql
            AS $$
            DECLARE
                chunk_skew_max numeric := 0;
                size_skew_max numeric := 0;
                warning_count integer := 0;
                unbalanced_count integer := 0;
                hot_partitions integer := 0;
            BEGIN
                SELECT
                    COALESCE(MAX(chunk_skew), 0),
                    COALESCE(MAX(size_skew), 0),
                    SUM(CASE WHEN health_status = 'WARNING' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN health_status = 'UNBALANCED' THEN 1 ELSE 0 END)
                INTO chunk_skew_max, size_skew_max, warning_count, unbalanced_count
                FROM partition_health_summary;

                IF chunk_skew_max >= 0.5 THEN
                    RETURN QUERY SELECT 'chunk_distribution', chunk_skew_max, 'CRITICAL', 'Chunk distribution exceeds critical threshold';
                ELSIF chunk_skew_max >= 0.3 THEN
                    RETURN QUERY SELECT 'chunk_distribution', chunk_skew_max, 'WARNING', 'Chunk distribution exceeds warning threshold';
                ELSE
                    RETURN QUERY SELECT 'chunk_distribution', chunk_skew_max, 'NORMAL', 'Chunk distribution within expected range';
                END IF;

                IF size_skew_max >= 0.5 THEN
                    RETURN QUERY SELECT 'size_distribution', size_skew_max, 'CRITICAL', 'Partition size skew exceeds critical threshold';
                ELSIF size_skew_max >= 0.3 THEN
                    RETURN QUERY SELECT 'size_distribution', size_skew_max, 'WARNING', 'Partition size skew exceeds warning threshold';
                ELSE
                    RETURN QUERY SELECT 'size_distribution', size_skew_max, 'NORMAL', 'Partition size skew within expected range';
                END IF;

                IF unbalanced_count > 0 THEN
                    RETURN QUERY SELECT 'partition_health', unbalanced_count::numeric, 'CRITICAL', format('%s partitions require rebalancing', unbalanced_count);
                ELSIF warning_count > 0 THEN
                    RETURN QUERY SELECT 'partition_health', warning_count::numeric, 'WARNING', format('%s partitions show early warnings', warning_count);
                ELSE
                    RETURN QUERY SELECT 'partition_health', 0::numeric, 'NORMAL', 'All partitions report healthy status';
                END IF;

                SELECT COUNT(*)
                INTO hot_partitions
                FROM partition_hot_spots
                WHERE chunks_last_hour > 0
                  AND hour_percentage >= 12.5;

                IF hot_partitions > 0 THEN
                    RETURN QUERY SELECT 'hot_partitions', hot_partitions::numeric, 'WARNING', 'High recent write activity detected';
                ELSE
                    RETURN QUERY SELECT 'hot_partitions', 0::numeric, 'NORMAL', 'No abnormal recent write activity detected';
                END IF;

                RETURN;
            END;
            $$;
            """
        )
    )


def downgrade() -> None:
    """Drop partition monitoring views and helper function."""

    conn = op.get_bind()

    conn.execute(text("DROP FUNCTION IF EXISTS analyze_partition_skew()"))
    conn.execute(text("DROP VIEW IF EXISTS partition_health_summary"))
    conn.execute(text("DROP VIEW IF EXISTS partition_stats"))
    conn.execute(text("DROP VIEW IF EXISTS partition_hot_spots"))
    conn.execute(text("DROP VIEW IF EXISTS partition_size_distribution"))
    conn.execute(text("DROP VIEW IF EXISTS partition_chunk_distribution"))
