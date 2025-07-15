"""init_collections_schema

Revision ID: b6af1f8a14e8
Revises: 860fb2e922f2
Create Date: 2025-07-15 15:25:43.925650

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b6af1f8a14e8"
down_revision: str | Sequence[str] | None = "860fb2e922f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema to collections-based architecture."""

    # Create new tables

    # 1. Create collections table
    op.create_table(
        "collections",
        sa.Column("id", sa.String(), nullable=False),  # Using String for UUID compatibility
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("vector_store_name", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False, server_default="1000"),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False, server_default="200"),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("vector_store_name"),
    )
    op.create_index(op.f("ix_collections_name"), "collections", ["name"], unique=True)
    op.create_index(op.f("ix_collections_owner_id"), "collections", ["owner_id"], unique=False)
    op.create_index(op.f("ix_collections_is_public"), "collections", ["is_public"], unique=False)

    # 2. Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), nullable=False),  # Using String for UUID compatibility
        sa.Column("collection_id", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "completed", "failed", name="document_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_collection_id"), "documents", ["collection_id"], unique=False)
    op.create_index(op.f("ix_documents_content_hash"), "documents", ["content_hash"], unique=False)
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)
    op.create_index("ix_documents_collection_content_hash", "documents", ["collection_id", "content_hash"], unique=True)

    # 3. Create api_keys table
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(), nullable=False),  # Using String for UUID compatibility
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("key_hash", sa.String(), nullable=False),
        sa.Column("permissions", sa.JSON(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_api_keys_key_hash"), "api_keys", ["key_hash"], unique=True)
    op.create_index(op.f("ix_api_keys_user_id"), "api_keys", ["user_id"], unique=False)
    op.create_index(op.f("ix_api_keys_is_active"), "api_keys", ["is_active"], unique=False)

    # 4. Create collection_permissions table
    op.create_table(
        "collection_permissions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("collection_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("api_key_id", sa.String(), nullable=True),
        sa.Column("permission", sa.Enum("read", "write", "admin", name="permission_type"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "(user_id IS NOT NULL AND api_key_id IS NULL) OR (user_id IS NULL AND api_key_id IS NOT NULL)",
            name="check_user_or_api_key",
        ),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_collection_permissions_collection_id"), "collection_permissions", ["collection_id"], unique=False
    )
    op.create_index(op.f("ix_collection_permissions_user_id"), "collection_permissions", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_collection_permissions_api_key_id"), "collection_permissions", ["api_key_id"], unique=False
    )
    # Unique constraint to prevent duplicate permissions
    op.create_index(
        "ix_collection_permissions_unique_user",
        "collection_permissions",
        ["collection_id", "user_id"],
        unique=True,
        postgresql_where=sa.text("user_id IS NOT NULL"),
    )
    op.create_index(
        "ix_collection_permissions_unique_api_key",
        "collection_permissions",
        ["collection_id", "api_key_id"],
        unique=True,
        postgresql_where=sa.text("api_key_id IS NOT NULL"),
    )

    # Add new columns to users table
    op.add_column("users", sa.Column("is_superuser", sa.Boolean(), nullable=False, server_default="0"))
    op.add_column("users", sa.Column("updated_at", sa.DateTime(), nullable=True))

    # Drop old tables
    # Note: We're dropping these tables which will lose all existing job and file data
    # This is a breaking change and should be communicated clearly

    # Drop indexes on files table first
    op.drop_index("idx_files_status", table_name="files")
    op.drop_index("idx_files_job_id", table_name="files")
    op.drop_index("idx_files_job_content_hash", table_name="files")
    op.drop_index("idx_files_doc_id", table_name="files")
    op.drop_index("idx_files_content_hash", table_name="files")

    # Drop tables
    op.drop_table("files")
    op.drop_table("jobs")


def downgrade() -> None:
    """Downgrade back to job-based architecture."""

    # Drop new columns from users table
    op.drop_column("users", "updated_at")
    op.drop_column("users", "is_superuser")

    # Recreate old tables

    # Recreate jobs table
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("directory_path", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=True),
        sa.Column("chunk_overlap", sa.Integer(), nullable=True),
        sa.Column("batch_size", sa.Integer(), nullable=True),
        sa.Column("vector_dim", sa.Integer(), nullable=True),
        sa.Column("quantization", sa.String(), nullable=True, server_default="float32"),
        sa.Column("instruction", sa.Text(), nullable=True),
        sa.Column("total_files", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("processed_files", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("failed_files", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("current_file", sa.String(), nullable=True),
        sa.Column("start_time", sa.String(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("parent_job_id", sa.String(), nullable=True),
        sa.Column("mode", sa.String(), nullable=True, server_default="create"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Recreate files table
    op.create_table(
        "files",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("modified", sa.String(), nullable=False),
        sa.Column("extension", sa.String(), nullable=False),
        sa.Column("hash", sa.String(), nullable=True),
        sa.Column("doc_id", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("chunks_created", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("vectors_created", sa.Integer(), nullable=True, server_default="0"),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Recreate indexes on files table
    op.create_index("idx_files_job_id", "files", ["job_id"], unique=False)
    op.create_index("idx_files_status", "files", ["status"], unique=False)
    op.create_index("idx_files_doc_id", "files", ["doc_id"], unique=False)
    op.create_index("idx_files_content_hash", "files", ["content_hash"], unique=False)
    op.create_index("idx_files_job_content_hash", "files", ["job_id", "content_hash"], unique=False)

    # Drop new tables

    # Drop indexes first
    op.drop_index("ix_collection_permissions_unique_api_key", table_name="collection_permissions")
    op.drop_index("ix_collection_permissions_unique_user", table_name="collection_permissions")
    op.drop_index(op.f("ix_collection_permissions_api_key_id"), table_name="collection_permissions")
    op.drop_index(op.f("ix_collection_permissions_user_id"), table_name="collection_permissions")
    op.drop_index(op.f("ix_collection_permissions_collection_id"), table_name="collection_permissions")
    op.drop_table("collection_permissions")

    op.drop_index(op.f("ix_api_keys_is_active"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_user_id"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_key_hash"), table_name="api_keys")
    op.drop_table("api_keys")

    op.drop_index("ix_documents_collection_content_hash", table_name="documents")
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_index(op.f("ix_documents_content_hash"), table_name="documents")
    op.drop_index(op.f("ix_documents_collection_id"), table_name="documents")
    op.drop_table("documents")

    op.drop_index(op.f("ix_collections_is_public"), table_name="collections")
    op.drop_index(op.f("ix_collections_owner_id"), table_name="collections")
    op.drop_index(op.f("ix_collections_name"), table_name="collections")
    op.drop_table("collections")

    # Note: SQLite doesn't support CREATE TYPE, so no need to drop types
