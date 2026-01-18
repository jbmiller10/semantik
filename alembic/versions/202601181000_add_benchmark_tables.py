"""Add benchmark tables for evaluation framework.

Revision ID: 202601181000
Revises: 202601140005
Create Date: 2026-01-18

Creates tables for the benchmarking feature:
- benchmark_datasets: Ground truth dataset storage
- benchmark_dataset_mappings: Dataset-to-collection bindings
- benchmark_queries: Individual queries within datasets
- benchmark_relevance: Query-document relevance judgments
- benchmarks: Benchmark definitions with config matrix
- benchmark_runs: Individual configuration run results
- benchmark_run_metrics: Aggregate metrics per run
- benchmark_query_results: Per-query detailed results
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202601181000"
down_revision = "202601140005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create benchmark tables."""
    # 1. benchmark_datasets - Ground truth dataset storage
    op.create_table(
        "benchmark_datasets",
        sa.Column("id", sa.String(), primary_key=True),  # UUID
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("query_count", sa.Integer(), nullable=False, default=0),
        sa.Column("raw_file_path", sa.String(512), nullable=True),
        sa.Column("schema_version", sa.String(32), nullable=False, default="1.0"),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_benchmark_datasets_owner_id", "benchmark_datasets", ["owner_id"])
    op.create_index("ix_benchmark_datasets_name", "benchmark_datasets", ["name"])

    # 2. benchmark_dataset_mappings - Dataset-to-collection bindings
    op.create_table(
        "benchmark_dataset_mappings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "dataset_id",
            sa.String(),
            sa.ForeignKey("benchmark_datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "collection_id",
            sa.String(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("mapping_status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("mapped_count", sa.Integer(), nullable=False, default=0),
        sa.Column("total_count", sa.Integer(), nullable=False, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_benchmark_dataset_mappings_dataset_id", "benchmark_dataset_mappings", ["dataset_id"])
    op.create_index("ix_benchmark_dataset_mappings_collection_id", "benchmark_dataset_mappings", ["collection_id"])
    op.create_unique_constraint(
        "uq_benchmark_dataset_mappings_dataset_collection",
        "benchmark_dataset_mappings",
        ["dataset_id", "collection_id"],
    )
    op.create_check_constraint(
        "ck_benchmark_dataset_mappings_status",
        "benchmark_dataset_mappings",
        "mapping_status IN ('pending', 'resolved', 'partial')",
    )

    # 3. benchmark_queries - Individual queries within datasets
    op.create_table(
        "benchmark_queries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "dataset_id",
            sa.String(),
            sa.ForeignKey("benchmark_datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_key", sa.String(255), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_metadata", sa.JSON(), nullable=True),
    )
    op.create_index("ix_benchmark_queries_dataset_id", "benchmark_queries", ["dataset_id"])
    op.create_unique_constraint(
        "uq_benchmark_queries_dataset_key",
        "benchmark_queries",
        ["dataset_id", "query_key"],
    )

    # 4. benchmark_relevance - Query-document relevance judgments
    op.create_table(
        "benchmark_relevance",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "benchmark_query_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_queries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "mapping_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_dataset_mappings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("doc_ref_hash", sa.String(64), nullable=False),
        sa.Column("doc_ref", sa.JSON(), nullable=False),
        sa.Column(
            "resolved_document_id",
            sa.String(),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("relevance_grade", sa.Integer(), nullable=False),
        sa.Column("relevance_metadata", sa.JSON(), nullable=True),
    )
    op.create_index("ix_benchmark_relevance_query_id", "benchmark_relevance", ["benchmark_query_id"])
    op.create_index("ix_benchmark_relevance_mapping_id", "benchmark_relevance", ["mapping_id"])
    op.create_index("ix_benchmark_relevance_doc_ref_hash", "benchmark_relevance", ["doc_ref_hash"])
    op.create_index(
        "ix_benchmark_relevance_resolved_document_id",
        "benchmark_relevance",
        ["resolved_document_id"],
    )
    op.create_check_constraint(
        "ck_benchmark_relevance_grade",
        "benchmark_relevance",
        "relevance_grade >= 0 AND relevance_grade <= 3",
    )

    # 5. benchmarks - Benchmark definitions with config matrix
    op.create_table(
        "benchmarks",
        sa.Column("id", sa.String(), primary_key=True),  # UUID
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "mapping_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_dataset_mappings.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "operation_uuid",
            sa.String(),
            sa.ForeignKey("operations.uuid", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("evaluation_unit", sa.String(32), nullable=False, server_default="query"),
        sa.Column("config_matrix", sa.JSON(), nullable=False),
        sa.Column("config_matrix_hash", sa.String(64), nullable=False),
        sa.Column("limits", sa.JSON(), nullable=True),
        sa.Column("collection_snapshot_hash", sa.String(64), nullable=True),
        sa.Column("reproducibility_metadata", sa.JSON(), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("metrics_to_compute", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("total_runs", sa.Integer(), nullable=False, default=0),
        sa.Column("completed_runs", sa.Integer(), nullable=False, default=0),
        sa.Column("failed_runs", sa.Integer(), nullable=False, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_benchmarks_owner_id", "benchmarks", ["owner_id"])
    op.create_index("ix_benchmarks_mapping_id", "benchmarks", ["mapping_id"])
    op.create_index("ix_benchmarks_status", "benchmarks", ["status"])
    op.create_index("ix_benchmarks_owner_status", "benchmarks", ["owner_id", "status"])
    op.create_check_constraint(
        "ck_benchmarks_status",
        "benchmarks",
        "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_benchmarks_top_k",
        "benchmarks",
        "top_k >= 1 AND top_k <= 100",
    )

    # 6. benchmark_runs - Individual configuration run results
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(), primary_key=True),  # UUID
        sa.Column(
            "benchmark_id",
            sa.String(),
            sa.ForeignKey("benchmarks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_order", sa.Integer(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("status_message", sa.Text(), nullable=True),
        sa.Column("indexing_duration_ms", sa.Integer(), nullable=True),
        sa.Column("evaluation_duration_ms", sa.Integer(), nullable=True),
        sa.Column("total_duration_ms", sa.Integer(), nullable=True),
        sa.Column("dense_collection_name", sa.String(255), nullable=True),
        sa.Column("sparse_collection_name", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("ix_benchmark_runs_benchmark_id", "benchmark_runs", ["benchmark_id"])
    op.create_index("ix_benchmark_runs_status", "benchmark_runs", ["status"])
    op.create_index("ix_benchmark_runs_benchmark_status", "benchmark_runs", ["benchmark_id", "status"])
    op.create_unique_constraint(
        "uq_benchmark_runs_benchmark_order",
        "benchmark_runs",
        ["benchmark_id", "run_order"],
    )
    op.create_check_constraint(
        "ck_benchmark_runs_status",
        "benchmark_runs",
        "status IN ('pending', 'indexing', 'evaluating', 'completed', 'failed')",
    )

    # 7. benchmark_run_metrics - Aggregate metrics per run
    op.create_table(
        "benchmark_run_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(),
            sa.ForeignKey("benchmark_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("k_value", sa.Integer(), nullable=True),
        sa.Column("metric_value", sa.Float(), nullable=False),
    )
    op.create_index("ix_benchmark_run_metrics_run_id", "benchmark_run_metrics", ["run_id"])
    op.create_unique_constraint(
        "uq_benchmark_run_metrics_run_metric_k",
        "benchmark_run_metrics",
        ["run_id", "metric_name", "k_value"],
    )

    # 8. benchmark_query_results - Per-query detailed results
    op.create_table(
        "benchmark_query_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(),
            sa.ForeignKey("benchmark_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "benchmark_query_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_queries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("retrieved_doc_ids", sa.JSON(), nullable=False),
        sa.Column("retrieved_debug", sa.JSON(), nullable=True),
        sa.Column("precision_at_k", sa.Float(), nullable=True),
        sa.Column("recall_at_k", sa.Float(), nullable=True),
        sa.Column("reciprocal_rank", sa.Float(), nullable=True),
        sa.Column("ndcg_at_k", sa.Float(), nullable=True),
        sa.Column("search_time_ms", sa.Integer(), nullable=True),
        sa.Column("rerank_time_ms", sa.Integer(), nullable=True),
    )
    op.create_index("ix_benchmark_query_results_run_id", "benchmark_query_results", ["run_id"])
    op.create_index("ix_benchmark_query_results_query_id", "benchmark_query_results", ["benchmark_query_id"])
    op.create_unique_constraint(
        "uq_benchmark_query_results_run_query",
        "benchmark_query_results",
        ["run_id", "benchmark_query_id"],
    )


def downgrade() -> None:
    """Remove benchmark tables."""
    # Drop in reverse order to respect FK dependencies
    op.drop_table("benchmark_query_results")
    op.drop_table("benchmark_run_metrics")
    op.drop_table("benchmark_runs")
    op.drop_table("benchmarks")
    op.drop_table("benchmark_relevance")
    op.drop_table("benchmark_queries")
    op.drop_table("benchmark_dataset_mappings")
    op.drop_table("benchmark_datasets")
