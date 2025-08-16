"""
Authentication mocking infrastructure for integration tests.

This module provides reusable authentication mocking utilities that ensure
consistent user representation across all integration tests. It handles
dependency overrides for FastAPI authentication and provides helper functions
for managing test users.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from fastapi import FastAPI
from shared.database.models import User
from sqlalchemy.ext.asyncio import AsyncSession
from webui.auth import get_password_hash


class TestUser:
    """
    Represents a test user with consistent attributes across tests.
    
    This class ensures that user data is consistent between database records
    and authentication contexts, preventing AccessDeniedError issues.
    """
    
    def __init__(
        self,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: str = "test_password",
        full_name: str = "Test User",
        is_active: bool = True,
        is_superuser: bool = False,
    ):
        """Initialize a test user with default or provided values."""
        self.id = user_id if user_id is not None else 1
        self.username = username or f"test_user_{uuid.uuid4().hex[:8]}"
        self.email = email or f"test_{uuid.uuid4().hex[:8]}@example.com"
        self.password = password
        self.full_name = full_name
        self.is_active = is_active
        self.is_superuser = is_superuser
        self.created_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.last_login = datetime.now(UTC)
        
    def to_db_model(self) -> User:
        """Convert to SQLAlchemy User model for database insertion."""
        return User(
            id=self.id,
            username=self.username,
            email=self.email,
            full_name=self.full_name,
            hashed_password=get_password_hash(self.password),
            is_active=self.is_active,
            is_superuser=self.is_superuser,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_login=self.last_login,
        )
    
    def to_auth_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format expected by authentication system.
        
        This format is what get_current_user returns and what services expect.
        """
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }
    
    def to_test_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format used in test fixtures.
        
        Includes the plain password for use in authentication tests.
        """
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "password": self.password,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
        }


class AuthMocker:
    """
    Manages authentication mocking for FastAPI applications.
    
    This class provides methods to override FastAPI dependencies for
    authentication, ensuring consistent user context in tests.
    """
    
    def __init__(self, app: FastAPI):
        """Initialize the auth mocker with a FastAPI application."""
        self.app = app
        self.current_user: Optional[TestUser] = None
        self._original_overrides = {}
        
    def set_user(self, user: TestUser) -> None:
        """Set the current authenticated user for mocking."""
        self.current_user = user
        
    def override_auth(self) -> None:
        """Override the get_current_user dependency with mock user."""
        from webui.auth import get_current_user
        
        if self.current_user is None:
            raise ValueError("No user set for authentication mocking")
        
        # Store original override if exists
        if get_current_user in self.app.dependency_overrides:
            self._original_overrides[get_current_user] = self.app.dependency_overrides[get_current_user]
        
        # Create override function that returns the mock user
        async def mock_get_current_user():
            return self.current_user.to_auth_dict()
        
        self.app.dependency_overrides[get_current_user] = mock_get_current_user
        
    def clear_auth(self) -> None:
        """Clear authentication override and restore original if exists."""
        from webui.auth import get_current_user
        
        if get_current_user in self._original_overrides:
            self.app.dependency_overrides[get_current_user] = self._original_overrides[get_current_user]
            del self._original_overrides[get_current_user]
        elif get_current_user in self.app.dependency_overrides:
            del self.app.dependency_overrides[get_current_user]
            
        self.current_user = None
        
    def __enter__(self):
        """Context manager entry - override auth."""
        self.override_auth()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear auth override."""
        self.clear_auth()


# Helper functions for common test scenarios

async def create_test_user(
    session: AsyncSession,
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
    is_superuser: bool = False,
) -> TestUser:
    """
    Create a test user in the database and return TestUser instance.
    
    Args:
        session: AsyncSession for database operations
        user_id: Optional specific user ID
        username: Optional specific username
        email: Optional specific email
        is_superuser: Whether user should be superuser
        
    Returns:
        TestUser instance with the created user's data
    """
    test_user = TestUser(
        user_id=user_id,
        username=username,
        email=email,
        is_superuser=is_superuser,
    )
    
    db_user = test_user.to_db_model()
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    
    # Update test_user with actual database values
    test_user.id = db_user.id
    test_user.created_at = db_user.created_at
    test_user.updated_at = db_user.updated_at
    
    return test_user


async def create_admin_user(
    session: AsyncSession,
    user_id: Optional[int] = None,
) -> TestUser:
    """
    Create an admin test user in the database.
    
    Args:
        session: AsyncSession for database operations
        user_id: Optional specific user ID
        
    Returns:
        TestUser instance with admin privileges
    """
    return await create_test_user(
        session,
        user_id=user_id,
        username=f"admin_{uuid.uuid4().hex[:8]}",
        email=f"admin_{uuid.uuid4().hex[:8]}@example.com",
        is_superuser=True,
    )


def mock_authenticated_user(app: FastAPI, user: TestUser) -> AuthMocker:
    """
    Create an AuthMocker with a specific user for testing.
    
    This is a convenience function for quickly setting up auth mocking.
    
    Args:
        app: FastAPI application instance
        user: TestUser to use for authentication
        
    Returns:
        Configured AuthMocker instance
    """
    mocker = AuthMocker(app)
    mocker.set_user(user)
    mocker.override_auth()
    return mocker


def create_default_test_user() -> TestUser:
    """
    Create a default test user for simple test cases.
    
    Returns:
        TestUser with default settings
    """
    return TestUser(
        user_id=1,
        username="default_test_user",
        email="test@example.com",
        full_name="Default Test User",
    )


def create_default_admin_user() -> TestUser:
    """
    Create a default admin user for admin test cases.
    
    Returns:
        TestUser with admin privileges
    """
    return TestUser(
        user_id=2,
        username="default_admin_user",
        email="admin@example.com",
        full_name="Default Admin User",
        is_superuser=True,
    )