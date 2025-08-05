"""add_partition_monitoring_views

Revision ID: 8f67aa430c5d
Revises: 6596eda04faa
Create Date: 2025-08-04 16:06:00.340778

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f67aa430c5d"
down_revision: str | Sequence[str] | None = "6596eda04faa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create view for partition size distribution
    op.execute(
        """
        CREATE OR REPLACE VIEW partition_size_distribution AS
        WITH partition_info AS (
            SELECT
                pt.tablename AS partition_name,
                pt.schemaname,
                SUBSTRING(pt.tablename FROM 'chunks_(\\d+)$')::INT AS partition_num,
                pg_size_pretty(pg_total_relation_size(pt.schemaname||'.'||pt.tablename)) as size_pretty,
                pg_total_relation_size(pt.schemaname||'.'||pt.tablename) as size_bytes,
                (SELECT COUNT(*) FROM pg_catalog.pg_stat_user_tables pst
                 WHERE pst.schemaname = pt.schemaname AND pst.relname = pt.tablename) as row_estimate
            FROM pg_catalog.pg_tables pt
            WHERE pt.tablename LIKE 'chunks_%'
            AND pt.tablename ~ 'chunks_[0-9]+$'
            AND pt.schemaname = 'public'
        )
        SELECT
            partition_num,
            partition_name,
            size_pretty,
            size_bytes,
            ROUND((size_bytes::NUMERIC / NULLIF(SUM(size_bytes) OVER (), 0) * 100), 2) as size_percentage,
            row_estimate
        FROM partition_info
        ORDER BY partition_num;
    """
    )

    # Create view for partition chunk count
    op.execute(
        """
        CREATE OR REPLACE VIEW partition_chunk_distribution AS
        WITH chunk_counts AS (
            SELECT
                tableoid::regclass::text AS partition_name,
                SUBSTRING(tableoid::regclass::text FROM 'chunks_(\\d+)$')::INT AS partition_num,
                COUNT(*) as chunk_count,
                COUNT(DISTINCT document_id) as document_count,
                COUNT(DISTINCT collection_id) as collection_count,
                AVG(token_count)::NUMERIC(10,2) as avg_tokens,
                MIN(created_at) as oldest_chunk,
                MAX(created_at) as newest_chunk
            FROM chunks
            GROUP BY tableoid
        )
        SELECT
            partition_num,
            partition_name,
            chunk_count,
            document_count,
            collection_count,
            avg_tokens,
            ROUND((chunk_count::NUMERIC / NULLIF(SUM(chunk_count) OVER (), 0) * 100), 2) as chunk_percentage,
            oldest_chunk,
            newest_chunk,
            EXTRACT(EPOCH FROM (newest_chunk - oldest_chunk))/3600 as age_hours
        FROM chunk_counts
        ORDER BY partition_num;
    """
    )

    # Create view for partition hot spots (high activity partitions)
    op.execute(
        """
        CREATE OR REPLACE VIEW partition_hot_spots AS
        WITH recent_activity AS (
            SELECT
                tableoid::regclass::text AS partition_name,
                SUBSTRING(tableoid::regclass::text FROM 'chunks_(\\d+)$')::INT AS partition_num,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as chunks_last_hour,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as chunks_last_day,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as chunks_last_week
            FROM chunks
            WHERE created_at > NOW() - INTERVAL '7 days'
            GROUP BY tableoid
        ),
        partition_totals AS (
            SELECT
                SUM(chunks_last_hour) as total_last_hour,
                SUM(chunks_last_day) as total_last_day,
                SUM(chunks_last_week) as total_last_week
            FROM recent_activity
        )
        SELECT
            ra.partition_num,
            ra.partition_name,
            ra.chunks_last_hour,
            ra.chunks_last_day,
            ra.chunks_last_week,
            CASE
                WHEN pt.total_last_hour > 0 THEN
                    ROUND((ra.chunks_last_hour::NUMERIC / pt.total_last_hour * 100), 2)
                ELSE 0
            END as hour_percentage,
            CASE
                WHEN pt.total_last_day > 0 THEN
                    ROUND((ra.chunks_last_day::NUMERIC / pt.total_last_day * 100), 2)
                ELSE 0
            END as day_percentage,
            CASE
                WHEN pt.total_last_week > 0 THEN
                    ROUND((ra.chunks_last_week::NUMERIC / pt.total_last_week * 100), 2)
                ELSE 0
            END as week_percentage
        FROM recent_activity ra
        CROSS JOIN partition_totals pt
        WHERE ra.chunks_last_day > 0
        ORDER BY ra.chunks_last_hour DESC, ra.partition_num;
    """
    )

    # Create view for partition health summary
    op.execute(
        """
        CREATE OR REPLACE VIEW partition_health_summary AS
        WITH stats AS (
            SELECT
                pcd.partition_num,
                pcd.chunk_count,
                pcd.chunk_percentage,
                psd.size_bytes,
                psd.size_percentage,
                phs.chunks_last_day,
                phs.day_percentage,
                -- Calculate skew factor (1.0 = perfectly balanced)
                ABS(pcd.chunk_percentage - (100.0 / 16)) as chunk_skew,
                ABS(psd.size_percentage - (100.0 / 16)) as size_skew
            FROM partition_chunk_distribution pcd
            JOIN partition_size_distribution psd USING (partition_num)
            LEFT JOIN partition_hot_spots phs USING (partition_num)
        )
        SELECT
            partition_num,
            chunk_count,
            chunk_percentage,
            pg_size_pretty(size_bytes) as partition_size,
            size_percentage,
            COALESCE(chunks_last_day, 0) as chunks_last_day,
            COALESCE(day_percentage, 0) as day_activity_percentage,
            chunk_skew,
            size_skew,
            CASE
                WHEN chunk_skew > 10 OR size_skew > 10 THEN 'UNBALANCED'
                WHEN chunk_skew > 5 OR size_skew > 5 THEN 'WARNING'
                ELSE 'HEALTHY'
            END as health_status,
            CASE
                WHEN chunk_skew > 10 OR size_skew > 10 THEN
                    'Partition is significantly unbalanced. Consider rebalancing.'
                WHEN chunk_skew > 5 OR size_skew > 5 THEN
                    'Partition shows signs of imbalance. Monitor closely.'
                ELSE 'Partition is well balanced.'
            END as recommendation
        FROM stats
        ORDER BY chunk_skew DESC, partition_num;
    """
    )

    # Create a function to analyze partition skew
    op.execute(
        """
        CREATE OR REPLACE FUNCTION analyze_partition_skew()
        RETURNS TABLE (
            metric TEXT,
            value NUMERIC,
            status TEXT,
            details TEXT
        ) AS $$
        DECLARE
            v_max_chunk_pct NUMERIC;
            v_min_chunk_pct NUMERIC;
            v_stddev_chunks NUMERIC;
            v_expected_pct NUMERIC := 100.0 / 16;  -- Expected percentage per partition
        BEGIN
            -- Get chunk distribution stats
            SELECT MAX(chunk_percentage), MIN(chunk_percentage), STDDEV(chunk_percentage)
            INTO v_max_chunk_pct, v_min_chunk_pct, v_stddev_chunks
            FROM partition_chunk_distribution;

            -- Overall skew metric
            RETURN QUERY
            SELECT
                'Overall Skew Factor'::TEXT,
                ROUND((v_max_chunk_pct - v_min_chunk_pct), 2),
                CASE
                    WHEN (v_max_chunk_pct - v_min_chunk_pct) > 20 THEN 'CRITICAL'
                    WHEN (v_max_chunk_pct - v_min_chunk_pct) > 10 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END,
                FORMAT('Max: %s%%, Min: %s%%, Expected: %s%%',
                    v_max_chunk_pct, v_min_chunk_pct, ROUND(v_expected_pct, 2));

            -- Standard deviation metric
            RETURN QUERY
            SELECT
                'Distribution Variance'::TEXT,
                ROUND(v_stddev_chunks, 2),
                CASE
                    WHEN v_stddev_chunks > 5 THEN 'HIGH'
                    WHEN v_stddev_chunks > 2.5 THEN 'MODERATE'
                    ELSE 'LOW'
                END,
                'Standard deviation of chunk distribution across partitions';

            -- Hot partition detection
            RETURN QUERY
            SELECT
                'Hot Partitions'::TEXT,
                COUNT(*)::NUMERIC,
                CASE
                    WHEN COUNT(*) > 3 THEN 'WARNING'
                    WHEN COUNT(*) > 0 THEN 'INFO'
                    ELSE 'HEALTHY'
                END,
                FORMAT('%s partitions have >%s%% of daily activity',
                    COUNT(*), ROUND(v_expected_pct * 2, 0))
            FROM partition_hot_spots
            WHERE day_percentage > v_expected_pct * 2;

            RETURN;
        END;
        $$ LANGUAGE plpgsql;
    """
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop function
    op.execute("DROP FUNCTION IF EXISTS analyze_partition_skew()")

    # Drop views in reverse order
    op.execute("DROP VIEW IF EXISTS partition_health_summary")
    op.execute("DROP VIEW IF EXISTS partition_hot_spots")
    op.execute("DROP VIEW IF EXISTS partition_chunk_distribution")
    op.execute("DROP VIEW IF EXISTS partition_size_distribution")
