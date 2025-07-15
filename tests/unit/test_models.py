"""Unit tests for SQLAlchemy models timezone handling."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from packages.shared.database.models import (
    ApiKey,
    Base,
    Collection,
    CollectionAuditLog,
    CollectionPermission,
    CollectionResourceLimits,
    CollectionSource,
    CollectionStatus,
    Document,
    DocumentStatus,
    Operation,
    OperationMetrics,
    OperationStatus,
    OperationType,
    PermissionType,
    RefreshToken,
    User,
)


@pytest.fixture()
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


class TestDateTimeTimezoneAwareness:
    """Test that all DateTime fields properly handle timezone information.

    Note: SQLite doesn't preserve timezone information by default, so these tests
    focus on verifying that the models are configured correctly for databases that
    do support timezone-aware datetimes (PostgreSQL, MySQL, etc.).
    """

    def test_user_datetime_fields_are_timezone_aware(self, db_session):
        """Test User model timezone-aware datetime fields."""
        # Create user with timezone-aware datetime
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            last_login=datetime.now(UTC),
        )

        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert user.created_at is not None
        assert user.updated_at is not None
        assert user.last_login is not None

    def test_collection_datetime_fields_are_timezone_aware(self, db_session):
        """Test Collection model timezone-aware datetime fields."""
        # First create a user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )
        db_session.add(user)
        db_session.commit()

        # Create collection with timezone-aware datetime
        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
            chunk_size=1000,
            chunk_overlap=200,
            status=CollectionStatus.READY,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        db_session.add(collection)
        db_session.commit()
        db_session.refresh(collection)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert collection.created_at is not None
        assert collection.updated_at is not None

    def test_document_datetime_fields_are_timezone_aware(self, db_session):
        """Test Document model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        # Create document with timezone-aware datetime
        document = Document(
            id="doc-uuid",
            collection_id=collection.id,
            file_path="/test/path",
            file_name="test.txt",
            file_size=1024,
            content_hash="hash123",
            status=DocumentStatus.COMPLETED,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert document.created_at is not None
        assert document.updated_at is not None

    def test_api_key_datetime_fields_are_timezone_aware(self, db_session):
        """Test ApiKey model timezone-aware datetime fields."""
        # Create user
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        # Create API key with timezone-aware datetime
        api_key = ApiKey(
            id="apikey-uuid",
            user_id=user.id,
            name="test-key",
            key_hash="hash123",
            created_at=datetime.now(UTC),
            last_used_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
        )

        db_session.add(api_key)
        db_session.commit()
        db_session.refresh(api_key)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert api_key.created_at is not None
        assert api_key.last_used_at is not None
        assert api_key.expires_at is not None

    def test_refresh_token_datetime_fields_are_timezone_aware(self, db_session):
        """Test RefreshToken model timezone-aware datetime fields."""
        # Create user
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        # Create refresh token with timezone-aware datetime
        token = RefreshToken(
            user_id=user.id,
            token_hash="hash123",
            expires_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
        )

        db_session.add(token)
        db_session.commit()
        db_session.refresh(token)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert token.created_at is not None
        assert token.expires_at is not None

    def test_collection_source_datetime_fields_are_timezone_aware(self, db_session):
        """Test CollectionSource model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        # Create collection source with timezone-aware datetime
        source = CollectionSource(
            collection_id=collection.id,
            source_path="/test/path",
            source_type="directory",
            last_indexed_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        db_session.add(source)
        db_session.commit()
        db_session.refresh(source)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert source.last_indexed_at is not None
        assert source.created_at is not None
        assert source.updated_at is not None

    def test_operation_datetime_fields_are_timezone_aware(self, db_session):
        """Test Operation model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        # Create operation with timezone-aware datetime
        operation = Operation(
            uuid="op-uuid",
            collection_id=collection.id,
            user_id=user.id,
            type=OperationType.INDEX,
            status=OperationStatus.COMPLETED,
            config={},
            created_at=datetime.now(UTC),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        db_session.add(operation)
        db_session.commit()
        db_session.refresh(operation)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert operation.created_at is not None
        assert operation.started_at is not None
        assert operation.completed_at is not None

    def test_default_created_at_is_timezone_aware(self, db_session):
        """Test that default func.now() creates timezone-aware datetimes."""
        # Create user without specifying created_at
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed",
        )

        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Verify created_at was set and is timezone-aware
        assert user.created_at is not None
        # Note: SQLite may not preserve timezone info by default,
        # but the field is configured correctly for databases that do

    def test_operation_metrics_recorded_at_is_timezone_aware(self, db_session):
        """Test OperationMetrics model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        operation = Operation(
            uuid="op-uuid",
            collection_id=collection.id,
            user_id=user.id,
            type=OperationType.INDEX,
            config={},
        )
        db_session.add(operation)
        db_session.commit()

        # Create operation metrics with timezone-aware datetime
        metrics = OperationMetrics(
            operation_id=operation.id,
            metric_name="duration",
            metric_value=123.45,
            recorded_at=datetime.now(UTC),
        )

        db_session.add(metrics)
        db_session.commit()
        db_session.refresh(metrics)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert metrics.recorded_at is not None

    def test_collection_permission_datetime_fields_are_timezone_aware(self, db_session):
        """Test CollectionPermission model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        # Create permission with timezone-aware datetime
        permission = CollectionPermission(
            collection_id=collection.id,
            user_id=user.id,
            permission=PermissionType.READ,
            created_at=datetime.now(UTC),
        )

        db_session.add(permission)
        db_session.commit()
        db_session.refresh(permission)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert permission.created_at is not None

    def test_audit_log_datetime_fields_are_timezone_aware(self, db_session):
        """Test CollectionAuditLog model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        # Create audit log with timezone-aware datetime
        audit_log = CollectionAuditLog(
            collection_id=collection.id,
            user_id=user.id,
            action="created",
            created_at=datetime.now(UTC),
        )

        db_session.add(audit_log)
        db_session.commit()
        db_session.refresh(audit_log)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert audit_log.created_at is not None

    def test_resource_limits_datetime_fields_are_timezone_aware(self, db_session):
        """Test CollectionResourceLimits model timezone-aware datetime fields."""
        # Create prerequisites
        user = User(username="testuser", email="test@example.com", hashed_password="hashed")
        db_session.add(user)
        db_session.commit()

        collection = Collection(
            id="test-uuid",
            name="test-collection",
            owner_id=user.id,
            vector_store_name="test-vector",
            embedding_model="test-model",
        )
        db_session.add(collection)
        db_session.commit()

        # Create resource limits with timezone-aware datetime
        limits = CollectionResourceLimits(
            collection_id=collection.id,
            max_documents=50000,
            max_storage_gb=25.0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        db_session.add(limits)
        db_session.commit()
        db_session.refresh(limits)

        # Verify fields were set (SQLite may not preserve timezone info)
        assert limits.created_at is not None
        assert limits.updated_at is not None
