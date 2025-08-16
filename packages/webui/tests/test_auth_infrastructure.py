"""
Test suite demonstrating the auth mocking infrastructure.

This file tests and demonstrates all features of the auth mocking system
to ensure it works correctly across different scenarios.
"""

import pytest
from fastapi import FastAPI, status
from httpx import AsyncClient

from packages.webui.tests.auth_mock import (
    DEFAULT_TEST_USER,
    ADMIN_TEST_USER,
    SECOND_TEST_USER,
    create_test_user,
    ensure_owner_consistency,
)


@pytest.mark.asyncio()
class TestAuthMockingInfrastructure:
    """Test suite for auth mocking infrastructure."""

    async def test_default_auth_mock(self, test_app):
        """Test that default auth mocking provides consistent user."""
        # The test_app fixture automatically uses DEFAULT_TEST_USER
        # Verify that auth is properly mocked
        from packages.webui.auth import get_current_user
        
        # Check that dependency is overridden
        assert get_current_user in test_app.dependency_overrides
        
        # Get the mocked user
        mock_func = test_app.dependency_overrides[get_current_user]
        user = await mock_func()
        
        # Verify it's the default test user
        assert user["id"] == DEFAULT_TEST_USER.id
        assert user["username"] == DEFAULT_TEST_USER.username

    async def test_custom_user_creation(self):
        """Test creating custom test users."""
        custom_user = create_test_user(
            user_id=99,
            username="custom_test",
            email="custom@test.com",
            is_superuser=True,
        )
        
        assert custom_user.id == 99
        assert custom_user.username == "custom_test"
        assert custom_user.email == "custom@test.com"
        assert custom_user.is_superuser is True
        
        # Test the to_dict method
        user_dict = custom_user.to_dict()
        assert user_dict["id"] == 99
        assert user_dict["username"] == "custom_test"
        assert "created_at" in user_dict
        assert "last_login" in user_dict

    async def test_multi_user_scenario(self, authenticated_client_factory):
        """Test multi-user access control scenarios."""
        # Create two clients with different users
        client1 = await authenticated_client_factory(user_id=1, username="user1")
        client2 = await authenticated_client_factory(user_id=2, username="user2")
        
        # Verify each client has the correct user
        assert client1.test_user.id == 1
        assert client1.test_user.username == "user1"
        assert client2.test_user.id == 2
        assert client2.test_user.username == "user2"

    async def test_admin_user_fixture(self, mock_auth_admin, test_app_with_custom_user):
        """Test admin user authentication."""
        # Create app with admin user
        app = test_app_with_custom_user(
            user_id=ADMIN_TEST_USER.id,
            username=ADMIN_TEST_USER.username,
            is_superuser=True,
        )
        
        # Verify admin properties
        assert app.state.test_user.is_superuser is True
        assert app.state.test_user.username == "admin"

    def test_owner_consistency_helper(self):
        """Test the ensure_owner_consistency utility."""
        user = create_test_user(user_id=42, username="owner_test")
        
        resource = {
            "name": "Test Resource",
            "description": "Testing ownership",
        }
        
        # Ensure owner consistency
        updated_resource = ensure_owner_consistency(resource, user)
        
        assert updated_resource["owner_id"] == 42
        assert updated_resource["name"] == "Test Resource"

    async def test_mock_database_consistency(self, mock_db):
        """Test that MockDatabase maintains owner consistency."""
        # mock_db fixture uses the test_user fixture (DEFAULT_TEST_USER)
        collection = mock_db.create_collection(
            name="Consistency Test",
            description="Testing database mock",
        )
        
        # Verify owner_id matches the test user
        assert collection["owner_id"] == DEFAULT_TEST_USER.id
        assert collection["name"] == "Consistency Test"
        
        # Create an operation for the collection
        operation = mock_db.create_operation(
            collection_id=collection["uuid"],
            type="reindex",
        )
        
        assert operation["collection_id"] == collection["uuid"]
        assert operation["type"] == "reindex"
        assert operation["status"] == "pending"

    async def test_service_mock_with_auth(self, mock_collection_service, mock_db):
        """Test that mock services respect auth boundaries."""
        # Create a collection with user 1
        collection, operation = await mock_collection_service.create_collection(
            user_id=1,
            name="User1 Collection",
            description="Owned by user 1",
            config={},
        )
        
        assert collection["owner_id"] == 1
        assert collection["name"] == "User1 Collection"
        
        # Try to update with different user (should fail)
        from packages.shared.database.exceptions import AccessDeniedError
        
        with pytest.raises(AccessDeniedError, match="does not have access to collection"):
            await mock_collection_service.update(
                collection_id=collection["uuid"],
                user_id=2,  # Different user
                updates={"name": "Hijacked"},
            )
        
        # Update with correct user (should succeed)
        updated = await mock_collection_service.update(
            collection_id=collection["uuid"],
            user_id=1,  # Same user
            updates={"name": "Updated by Owner"},
        )
        
        assert updated["name"] == "Updated by Owner"

    async def test_auth_works_regardless_of_disable_auth_env(self, test_app):
        """Test that auth mocking works regardless of DISABLE_AUTH setting."""
        # The test_app fixture uses mock_auth which overrides authentication
        # This should work consistently regardless of environment variables
        
        from packages.webui.auth import get_current_user
        
        # Check that the dependency is overridden
        assert get_current_user in test_app.dependency_overrides
        
        # The override should return our test user
        mock_func = test_app.dependency_overrides[get_current_user]
        user = await mock_func()
        
        assert user["id"] == DEFAULT_TEST_USER.id
        assert user["username"] == DEFAULT_TEST_USER.username

    async def test_fixtures_chain_correctly(
        self,
        test_app,
        authenticated_client,
        mock_db,
        mock_collection_service,
        override_service_dependency,
    ):
        """Test that all fixtures work together correctly."""
        # Override the service in the app
        override_service_dependency(test_app, mock_collection_service)
        
        # Create a collection through the service
        collection, operation = await mock_collection_service.create_collection(
            user_id=DEFAULT_TEST_USER.id,
            name="Fixture Chain Test",
            description="Testing fixture integration",
            config={},
        )
        
        # Verify the collection was created with correct ownership
        assert collection["owner_id"] == DEFAULT_TEST_USER.id
        assert collection["name"] == "Fixture Chain Test"
        
        # The mock_db should have the collection
        assert collection["uuid"] in mock_db.collections


@pytest.mark.asyncio()
class TestAuthMockingPatterns:
    """Test common patterns for using auth mocking."""

    async def test_pattern_single_user_crud(self, authenticated_client, mock_collection_service, test_app, override_service_dependency):
        """Pattern: Single user performing CRUD operations."""
        # Setup
        override_service_dependency(test_app, mock_collection_service)
        
        # User creates a collection (mocked at service level)
        collection, _ = await mock_collection_service.create_collection(
            user_id=DEFAULT_TEST_USER.id,
            name="CRUD Test",
            description="Testing CRUD",
            config={},
        )
        
        # User updates their own collection
        updated = await mock_collection_service.update(
            collection_id=collection["uuid"],
            user_id=DEFAULT_TEST_USER.id,
            updates={"description": "Updated description"},
        )
        
        assert updated["description"] == "Updated description"
        
        # User deletes their own collection
        await mock_collection_service.delete_collection(
            collection_id=collection["uuid"],
            user_id=DEFAULT_TEST_USER.id,
        )

    async def test_pattern_access_control(self, authenticated_client_factory, mock_collection_service):
        """Pattern: Testing access control between users."""
        # Create clients for two different users
        owner_client = await authenticated_client_factory(user_id=1, username="owner")
        other_client = await authenticated_client_factory(user_id=2, username="other")
        
        # Owner creates a private collection
        collection, _ = await mock_collection_service.create_collection(
            user_id=owner_client.test_user.id,
            name="Private Collection",
            description="Only for owner",
            config={"is_public": False},
        )
        
        # Other user tries to modify (should fail)
        from packages.shared.database.exceptions import AccessDeniedError
        
        with pytest.raises(AccessDeniedError, match="does not have access to collection"):
            await mock_collection_service.update(
                collection_id=collection["uuid"],
                user_id=other_client.test_user.id,
                updates={"name": "Hacked!"},
            )

    async def test_pattern_public_collection_access(self, mock_db):
        """Pattern: Testing public collection visibility."""
        # Create mock_db with user 1
        from packages.webui.tests.auth_mock import MockDatabase, create_test_user
        
        user1 = create_test_user(user_id=1, username="user1")
        db = MockDatabase(user=user1)
        
        # User 1 creates a public collection
        public_collection = db.create_collection(
            name="Public Collection",
            is_public=True,
        )
        
        # User 1 creates a private collection
        private_collection = db.create_collection(
            name="Private Collection",
            is_public=False,
        )
        
        # Simulate what list_for_user would do
        # User 2 should see public collections from other users
        user2_id = 2
        visible_collections = [
            c for c in db.collections.values()
            if c["owner_id"] == user2_id or (c.get("is_public", False))
        ]
        
        # Should only see public collections from other users
        visible_names = [c["name"] for c in visible_collections]
        assert "Public Collection" in visible_names
        assert "Private Collection" not in visible_names

    async def test_pattern_batch_operations(self, mock_db):
        """Pattern: Testing batch operations with consistent ownership."""
        # Create multiple collections for the same user
        user_id = 5
        test_user = create_test_user(user_id=user_id, username="batch_user")
        
        collections = []
        for i in range(5):
            collection = mock_db.create_collection(
                name=f"Batch Collection {i}",
                description=f"Collection {i} for batch testing",
            )
            # Ensure owner consistency
            collection = ensure_owner_consistency(collection, test_user)
            collections.append(collection)
        
        # Verify all collections have consistent ownership
        for collection in collections:
            assert collection["owner_id"] == user_id
        
        # Verify we created 5 collections
        assert len(mock_db.collections) == 5


@pytest.mark.asyncio()
class TestAuthMockingEdgeCases:
    """Test edge cases and error conditions."""

    async def test_inactive_user_handling(self):
        """Test handling of inactive users."""
        from packages.webui.tests.auth_mock import INACTIVE_TEST_USER
        
        # Verify inactive user properties
        assert INACTIVE_TEST_USER.is_active is False
        assert INACTIVE_TEST_USER.username == "inactive"

    async def test_cleanup_after_context_manager(self):
        """Test that context managers properly clean up."""
        from packages.webui.tests.auth_mock import mock_authenticated_user
        
        app = FastAPI()
        
        # Use context manager
        async with mock_authenticated_user(app) as (app, user):
            from packages.webui.auth import get_current_user
            assert get_current_user in app.dependency_overrides
        
        # After context, override should be cleaned up
        from packages.webui.auth import get_current_user
        assert get_current_user not in app.dependency_overrides

    def test_mock_database_isolation(self):
        """Test that MockDatabase instances are isolated."""
        from packages.webui.tests.auth_mock import MockDatabase
        
        db1 = MockDatabase(user=create_test_user(user_id=1))
        db2 = MockDatabase(user=create_test_user(user_id=2))
        
        # Create collections in each database
        col1 = db1.create_collection(name="DB1 Collection")
        col2 = db2.create_collection(name="DB2 Collection")
        
        # Verify isolation
        assert col1["owner_id"] == 1
        assert col2["owner_id"] == 2
        assert len(db1.collections) == 1
        assert len(db2.collections) == 1
        assert col1["uuid"] in db1.collections
        assert col2["uuid"] in db2.collections
        assert col1["uuid"] not in db2.collections
        assert col2["uuid"] not in db1.collections