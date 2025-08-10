"""implement_100_direct_list_partitions

Revision ID: ae558c9e183f
Revises: add_chunking_strategy_cols
Create Date: 2025-08-10 02:20:06.337096

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'ae558c9e183f'
down_revision: Union[str, Sequence[str], None] = 'add_chunking_strategy_cols'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Implement 100 direct LIST partitions for optimal chunk distribution.
    
    This migration:
    1. Drops the old chunks table with 16 HASH partitions
    2. Creates a new chunks table with 100 LIST partitions
    3. Uses PostgreSQL's hashtext() for even distribution
    4. Creates monitoring views for partition health
    """
    
    conn = op.get_bind()
    
    # Step 1: Drop old tables and views (we're pre-release!)
    conn.execute(text("DROP VIEW IF EXISTS active_chunking_configs CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS refresh_collection_chunking_stats() CASCADE"))
    conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS partition_mappings CASCADE"))  # Remove any old mapping tables
    
    # Step 2: Create trigger function to compute partition key
    # We use a trigger because PostgreSQL doesn't allow expressions or generated columns
    # in partition keys when combined with PRIMARY KEY constraints
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION compute_partition_key()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.partition_key := mod(hashtext(NEW.collection_id::text), 100);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    # Step 3: Create new partitioned table with regular partition_key column
    conn.execute(text("""
        CREATE TABLE chunks (
            id BIGSERIAL,
            collection_id VARCHAR NOT NULL,
            partition_key INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            document_id VARCHAR,
            chunking_config_id INTEGER,
            start_offset INTEGER,
            end_offset INTEGER,
            token_count INTEGER,
            embedding_vector_id VARCHAR,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (id, collection_id, partition_key),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
        ) PARTITION BY LIST (partition_key)
    """))
    
    # Step 4: Create trigger to auto-compute partition_key on INSERT
    conn.execute(text("""
        CREATE TRIGGER set_partition_key
        BEFORE INSERT ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION compute_partition_key();
    """))
    
    # Step 5: Create 100 partitions with proper indexes
    conn.execute(text("""
        DO $$
        DECLARE
            i INT;
        BEGIN
            FOR i IN 0..99 LOOP
                -- Create partition
                EXECUTE format('
                    CREATE TABLE chunks_part_%s PARTITION OF chunks
                    FOR VALUES IN (%s)',
                    LPAD(i::text, 2, '0'),
                    i
                );
                
                -- Create index on collection_id for each partition
                EXECUTE format('
                    CREATE INDEX idx_chunks_part_%s_collection 
                    ON chunks_part_%s(collection_id)',
                    LPAD(i::text, 2, '0'),
                    LPAD(i::text, 2, '0')
                );
                
                -- Create index on created_at for each partition
                EXECUTE format('
                    CREATE INDEX idx_chunks_part_%s_created 
                    ON chunks_part_%s(created_at)',
                    LPAD(i::text, 2, '0'),
                    LPAD(i::text, 2, '0')
                );
                
                -- Create index on chunk_index for each partition
                EXECUTE format('
                    CREATE INDEX idx_chunks_part_%s_chunk_index 
                    ON chunks_part_%s(collection_id, chunk_index)',
                    LPAD(i::text, 2, '0'),
                    LPAD(i::text, 2, '0')
                );
                
                -- Create index on document_id if it exists
                EXECUTE format('
                    CREATE INDEX idx_chunks_part_%s_document 
                    ON chunks_part_%s(document_id)
                    WHERE document_id IS NOT NULL',
                    LPAD(i::text, 2, '0'),
                    LPAD(i::text, 2, '0')
                );
            END LOOP;
        END $$;
    """))
    
    # Step 6: Create monitoring views for partition health
    
    # Main health monitoring view
    conn.execute(text("""
        CREATE OR REPLACE VIEW partition_health AS
        WITH partition_stats AS (
            SELECT 
                schemaname,
                tablename as partition_name,
                SUBSTRING(tablename FROM 'chunks_part_([0-9]+)')::INT as partition_id,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                n_live_tup as row_count,
                n_dead_tup as dead_rows,
                last_vacuum,
                last_autovacuum,
                n_tup_ins as inserts_since_vacuum,
                n_tup_upd as updates_since_vacuum,
                n_tup_del as deletes_since_vacuum
            FROM pg_stat_user_tables
            WHERE tablename LIKE 'chunks_part_%'
        ),
        stats_summary AS (
            SELECT 
                AVG(row_count) as avg_rows,
                MAX(row_count) as max_rows,
                MIN(row_count) as min_rows,
                STDDEV(row_count) as stddev_rows,
                AVG(size_bytes) as avg_size,
                SUM(row_count) as total_rows,
                SUM(size_bytes) as total_size
            FROM partition_stats
        )
        SELECT 
            ps.*,
            pg_size_pretty(ps.size_bytes) as size_pretty,
            ROUND((ps.row_count::NUMERIC / NULLIF(ss.avg_rows, 0) - 1) * 100, 2) as pct_deviation_from_avg,
            CASE 
                WHEN ss.avg_rows > 0 AND ps.row_count > ss.avg_rows * 1.2 THEN 'HOT'
                WHEN ss.avg_rows > 0 AND ps.row_count < ss.avg_rows * 0.8 THEN 'COLD'
                ELSE 'NORMAL'
            END as partition_status,
            ps.dead_rows > ps.row_count * 0.1 as needs_vacuum
        FROM partition_stats ps
        CROSS JOIN stats_summary ss
        ORDER BY partition_id;
    """))
    
    # Distribution analysis view
    conn.execute(text("""
        CREATE OR REPLACE VIEW partition_distribution AS
        WITH partition_counts AS (
            SELECT 
                mod(hashtext(collection_id::text), 100) as partition_id,
                COUNT(DISTINCT collection_id) as collection_count,
                COUNT(*) as chunk_count
            FROM chunks
            GROUP BY mod(hashtext(collection_id::text), 100)
        ),
        distribution_stats AS (
            SELECT 
                COUNT(*) as partitions_used,
                AVG(chunk_count) as avg_chunks_per_partition,
                STDDEV(chunk_count) as stddev_chunks,
                MAX(chunk_count) as max_chunks,
                MIN(chunk_count) as min_chunks,
                MAX(chunk_count)::FLOAT / NULLIF(AVG(chunk_count), 0) as max_skew_ratio
            FROM partition_counts
        )
        SELECT 
            ds.*,
            CASE 
                WHEN max_skew_ratio > 1.2 THEN 'REBALANCE NEEDED'
                WHEN max_skew_ratio > 1.1 THEN 'WARNING'
                ELSE 'HEALTHY'
            END as distribution_status,
            100 - partitions_used as empty_partitions
        FROM distribution_stats ds;
    """))
    
    # Step 7: Create helper functions for partition assignment
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION get_partition_for_collection(collection_id VARCHAR)
        RETURNS TEXT AS $$
        BEGIN
            RETURN 'chunks_part_' || LPAD((mod(hashtext(collection_id::text), 100))::text, 2, '0');
        END;
        $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
        
        -- Also create a function to get the partition key directly
        CREATE OR REPLACE FUNCTION get_partition_key(collection_id VARCHAR)
        RETURNS INTEGER AS $$
        BEGIN
            RETURN mod(hashtext(collection_id::text), 100);
        END;
        $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
    """))
    
    # Step 8: Create function to analyze partition skew
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION analyze_partition_skew()
        RETURNS TABLE(
            status TEXT,
            avg_rows NUMERIC,
            max_rows BIGINT,
            min_rows BIGINT,
            max_skew_ratio NUMERIC,
            partitions_over_threshold INT,
            recommendation TEXT
        ) AS $$
        DECLARE
            v_avg_rows NUMERIC;
            v_max_rows BIGINT;
            v_min_rows BIGINT;
            v_max_skew NUMERIC;
            v_over_threshold INT;
        BEGIN
            -- Calculate statistics
            SELECT 
                AVG(n_live_tup)::NUMERIC,
                MAX(n_live_tup),
                MIN(n_live_tup)
            INTO v_avg_rows, v_max_rows, v_min_rows
            FROM pg_stat_user_tables
            WHERE tablename LIKE 'chunks_part_%';
            
            -- Calculate skew ratio
            v_max_skew := CASE 
                WHEN v_avg_rows > 0 THEN v_max_rows::NUMERIC / v_avg_rows
                ELSE 0
            END;
            
            -- Count partitions over 20% threshold
            SELECT COUNT(*)
            INTO v_over_threshold
            FROM pg_stat_user_tables
            WHERE tablename LIKE 'chunks_part_%'
              AND n_live_tup > v_avg_rows * 1.2;
            
            RETURN QUERY
            SELECT 
                CASE 
                    WHEN v_max_skew > 1.5 THEN 'CRITICAL'
                    WHEN v_max_skew > 1.2 THEN 'WARNING'
                    ELSE 'HEALTHY'
                END as status,
                ROUND(v_avg_rows, 2) as avg_rows,
                v_max_rows as max_rows,
                v_min_rows as min_rows,
                ROUND(v_max_skew, 3) as max_skew_ratio,
                v_over_threshold as partitions_over_threshold,
                CASE 
                    WHEN v_max_skew > 1.5 THEN 'Severe skew detected. Consider data redistribution strategy.'
                    WHEN v_max_skew > 1.2 THEN 'Moderate skew detected. Monitor closely.'
                    ELSE 'Distribution is healthy.'
                END as recommendation;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    # Step 9: Create active chunking configs view (recreate with new structure)
    conn.execute(text("""
        CREATE VIEW active_chunking_configs AS
        SELECT
            cc.id,
            cc.strategy_id,
            cc.config_hash,
            cc.config_data,
            cc.created_at,
            cc.use_count,
            cc.last_used_at,
            cs.name as strategy_name,
            sub.collections_using
        FROM chunking_configs cc
        JOIN chunking_strategies cs ON cc.strategy_id = cs.id
        LEFT JOIN (
            SELECT chunking_config_id, COUNT(DISTINCT collection_id) as collections_using
            FROM chunks
            WHERE chunking_config_id IS NOT NULL
            GROUP BY chunking_config_id
        ) sub ON cc.id = sub.chunking_config_id
        WHERE cc.use_count > 0;
    """))
    
    # Step 10: Create materialized view for collection statistics
    conn.execute(text("""
        CREATE MATERIALIZED VIEW collection_chunking_stats AS
        SELECT
            c.id,
            c.name,
            COUNT(DISTINCT ch.document_id) as chunked_documents,
            COUNT(ch.id) as total_chunks,
            AVG(ch.token_count)::NUMERIC(10,2) as avg_tokens_per_chunk,
            MAX(ch.created_at) as last_chunk_created,
            mod(hashtext(c.id::text), 100) as partition_id,
            get_partition_for_collection(c.id) as partition_name
        FROM collections c
        LEFT JOIN chunks ch ON c.id = ch.collection_id
        GROUP BY c.id, c.name
        WITH DATA;
    """))
    
    # Create index on materialized view
    conn.execute(text("""
        CREATE UNIQUE INDEX ix_collection_chunking_stats_id 
        ON collection_chunking_stats(id);
    """))
    
    # Create refresh function for materialized view
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION refresh_collection_chunking_stats()
        RETURNS void AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY collection_chunking_stats;
        END;
        $$ LANGUAGE plpgsql;
    """))


def downgrade() -> None:
    """
    Rollback to 16 HASH partitions (not recommended).
    """
    
    conn = op.get_bind()
    
    # Drop all new views and functions
    conn.execute(text("DROP FUNCTION IF EXISTS analyze_partition_skew() CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS get_partition_key(VARCHAR) CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS get_partition_for_collection(VARCHAR) CASCADE"))
    conn.execute(text("DROP VIEW IF EXISTS partition_distribution CASCADE"))
    conn.execute(text("DROP VIEW IF EXISTS partition_health CASCADE"))
    conn.execute(text("DROP VIEW IF EXISTS active_chunking_configs CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS refresh_collection_chunking_stats() CASCADE"))
    conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE"))
    
    # Drop the trigger and trigger function
    conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS compute_partition_key() CASCADE"))
    
    # Drop the 100-partition table
    conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
    
    # Recreate the original 16-partition structure
    conn.execute(text("""
        CREATE TABLE chunks (
            id UUID DEFAULT gen_random_uuid() NOT NULL,
            collection_id VARCHAR NOT NULL,
            document_id VARCHAR NOT NULL,
            chunking_config_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            token_count INTEGER,
            embedding_vector_id VARCHAR,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            meta JSON,
            PRIMARY KEY (id, collection_id),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
        ) PARTITION BY HASH (collection_id);
    """))
    
    # Create 16 partitions
    for i in range(16):
        conn.execute(text(f"""
            CREATE TABLE chunks_p{i} PARTITION OF chunks
            FOR VALUES WITH (MODULUS 16, REMAINDER {i});
        """))
    
    # Recreate original indexes
    conn.execute(text("""
        CREATE INDEX ix_chunks_collection_id_document_id ON chunks(collection_id, document_id);
        CREATE INDEX ix_chunks_document_id ON chunks(document_id);
        CREATE INDEX ix_chunks_chunking_config_id ON chunks(chunking_config_id);
        CREATE INDEX ix_chunks_collection_id_chunk_index ON chunks(collection_id, chunk_index);
        CREATE INDEX ix_chunks_created_at ON chunks(created_at);
    """))