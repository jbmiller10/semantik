"""Add operations and supporting tables for collections

Revision ID: 91784cc819aa
Revises: b6af1f8a14e8
Create Date: 2025-07-15 15:41:37.186829

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '91784cc819aa'
down_revision: Union[str, Sequence[str], None] = 'b6af1f8a14e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add operations, collection_sources, audit_log and supporting tables."""
    
    # Note: SQLite doesn't support CREATE TYPE, so we'll use CHECK constraints instead
    # The Enum columns below will work with SQLAlchemy's enum handling for SQLite
    
    # 1. Update collections table to add status fields
    op.add_column('collections', sa.Column('status', sa.Enum('pending', 'ready', 'processing', 'error', 'degraded', name='collection_status'), nullable=False, server_default='pending'))
    op.add_column('collections', sa.Column('status_message', sa.Text(), nullable=True))
    op.add_column('collections', sa.Column('qdrant_collections', sa.JSON(), nullable=True))
    op.add_column('collections', sa.Column('qdrant_staging', sa.JSON(), nullable=True))
    op.add_column('collections', sa.Column('document_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('collections', sa.Column('vector_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('collections', sa.Column('total_size_bytes', sa.Integer(), nullable=False, server_default='0'))
    
    # 2. Create collection_sources table
    op.create_table(
        'collection_sources',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('collection_id', sa.String(), nullable=False),
        sa.Column('source_path', sa.String(), nullable=False),
        sa.Column('source_type', sa.String(), nullable=False, server_default='directory'),  # directory, file, url, etc.
        sa.Column('document_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('size_bytes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('collection_id', 'source_path', name='uq_collection_source_path')
    )
    op.create_index(op.f('ix_collection_sources_collection_id'), 'collection_sources', ['collection_id'], unique=False)
    
    # 3. Create operations table
    op.create_table(
        'operations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', sa.String(), nullable=False),  # For external reference
        sa.Column('collection_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('type', sa.Enum('index', 'append', 'reindex', 'remove_source', 'delete', name='operation_type'), nullable=False),
        sa.Column('status', sa.Enum('pending', 'processing', 'completed', 'failed', 'cancelled', name='operation_status'), nullable=False, server_default='pending'),
        sa.Column('task_id', sa.String(), nullable=True),  # Celery task ID
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid')
    )
    op.create_index(op.f('ix_operations_collection_id'), 'operations', ['collection_id'], unique=False)
    op.create_index(op.f('ix_operations_status'), 'operations', ['status'], unique=False)
    op.create_index(op.f('ix_operations_type'), 'operations', ['type'], unique=False)
    op.create_index(op.f('ix_operations_user_id'), 'operations', ['user_id'], unique=False)
    op.create_index(op.f('ix_operations_created_at'), 'operations', ['created_at'], unique=False)
    
    # 4. Create collection_audit_log table
    op.create_table(
        'collection_audit_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('collection_id', sa.String(), nullable=False),
        sa.Column('operation_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(), nullable=False),  # created, updated, deleted, reindexed, etc.
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['operation_id'], ['operations.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_collection_audit_log_collection_id'), 'collection_audit_log', ['collection_id'], unique=False)
    op.create_index(op.f('ix_collection_audit_log_user_id'), 'collection_audit_log', ['user_id'], unique=False)
    op.create_index(op.f('ix_collection_audit_log_created_at'), 'collection_audit_log', ['created_at'], unique=False)
    
    # 5. Create collection_resource_limits table
    op.create_table(
        'collection_resource_limits',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('collection_id', sa.String(), nullable=False),
        sa.Column('max_documents', sa.Integer(), nullable=True, server_default='100000'),
        sa.Column('max_storage_gb', sa.Float(), nullable=True, server_default='50.0'),
        sa.Column('max_operations_per_hour', sa.Integer(), nullable=True, server_default='10'),
        sa.Column('max_sources', sa.Integer(), nullable=True, server_default='10'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('collection_id')
    )
    
    # 6. Create operation_metrics table
    op.create_table(
        'operation_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('operation_id', sa.Integer(), nullable=False),
        sa.Column('metric_name', sa.String(), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['operation_id'], ['operations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_operation_metrics_operation_id'), 'operation_metrics', ['operation_id'], unique=False)
    op.create_index(op.f('ix_operation_metrics_recorded_at'), 'operation_metrics', ['recorded_at'], unique=False)
    
    # 7. Update documents table to add source reference
    # Using batch operations for SQLite compatibility
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_documents_source', 'collection_sources', ['source_id'], ['id'])
        batch_op.create_index(batch_op.f('ix_documents_source_id'), ['source_id'], unique=False)


def downgrade() -> None:
    """Remove operations and supporting tables."""
    
    # Drop indexes and foreign keys from documents table
    # Using batch operations for SQLite compatibility
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_documents_source_id'))
        batch_op.drop_constraint('fk_documents_source', type_='foreignkey')
        batch_op.drop_column('source_id')
    
    # Drop tables in reverse order of creation
    op.drop_index(op.f('ix_operation_metrics_recorded_at'), table_name='operation_metrics')
    op.drop_index(op.f('ix_operation_metrics_operation_id'), table_name='operation_metrics')
    op.drop_table('operation_metrics')
    
    op.drop_table('collection_resource_limits')
    
    op.drop_index(op.f('ix_collection_audit_log_created_at'), table_name='collection_audit_log')
    op.drop_index(op.f('ix_collection_audit_log_user_id'), table_name='collection_audit_log')
    op.drop_index(op.f('ix_collection_audit_log_collection_id'), table_name='collection_audit_log')
    op.drop_table('collection_audit_log')
    
    op.drop_index(op.f('ix_operations_created_at'), table_name='operations')
    op.drop_index(op.f('ix_operations_user_id'), table_name='operations')
    op.drop_index(op.f('ix_operations_type'), table_name='operations')
    op.drop_index(op.f('ix_operations_status'), table_name='operations')
    op.drop_index(op.f('ix_operations_collection_id'), table_name='operations')
    op.drop_table('operations')
    
    op.drop_index(op.f('ix_collection_sources_collection_id'), table_name='collection_sources')
    op.drop_table('collection_sources')
    
    # Drop columns from collections table
    op.drop_column('collections', 'total_size_bytes')
    op.drop_column('collections', 'vector_count')
    op.drop_column('collections', 'document_count')
    op.drop_column('collections', 'qdrant_staging')
    op.drop_column('collections', 'qdrant_collections')
    op.drop_column('collections', 'status_message')
    op.drop_column('collections', 'status')
    
    # Note: No need to drop types in SQLite as they are handled differently
