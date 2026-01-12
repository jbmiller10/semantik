"""PostgreSQL implementation of UserRepository."""

import logging
from datetime import UTC, datetime
from typing import Any

from passlib.context import CryptContext
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.base import UserRepository
from shared.database.exceptions import DatabaseOperationError, EntityAlreadyExistsError, InvalidUserIdError
from shared.database.models import User

from .base import PostgreSQLBaseRepository

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PostgreSQLUserRepository(PostgreSQLBaseRepository, UserRepository):
    """PostgreSQL implementation of UserRepository with optimized queries."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        super().__init__(session, User)
        self.pwd_context = pwd_context

    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user.

        Args:
            user_data: Dictionary containing user fields

        Returns:
            Created user as dictionary

        Raises:
            EntityAlreadyExistsError: If username or email already exists
            DatabaseOperationError: For database errors
        """
        try:
            # Validate required fields
            username = user_data.get("username")
            email = user_data.get("email")
            hashed_password = user_data.get("hashed_password")

            if not username or not email or not hashed_password:
                raise ValueError("Username, email, and hashed_password are required")

            # Check if username or email already exists
            existing = await self.session.scalar(
                select(User).where((User.username == username) | (User.email == email))
            )
            if existing:
                if existing.username == username:
                    raise EntityAlreadyExistsError("user", f"username: {username}")
                raise EntityAlreadyExistsError("user", f"email: {email}")

            # Create user with PostgreSQL RETURNING clause
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=user_data.get("full_name"),
                is_active=user_data.get("is_active", True),
                is_superuser=user_data.get("is_superuser", False),
            )

            self.session.add(user)
            await self.session.flush()

            # Refresh the user to get database-generated defaults (like created_at)
            await self.session.refresh(user)

            # Debug logging to understand datetime issue
            logger.debug(f"User after refresh - created_at type: {type(user.created_at)}, value: {user.created_at}")
            logger.debug(f"User after refresh - updated_at type: {type(user.updated_at)}, value: {user.updated_at}")

            logger.info(f"Created user {user.id} with username '{username}'")

            # Return user as dictionary
            result = self._user_to_dict(user)
            assert result is not None  # User was just created, can't be None
            return result

        except EntityAlreadyExistsError:
            raise
        except IntegrityError as e:
            self.handle_integrity_error(e, "create_user")
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise DatabaseOperationError("create", "user", str(e)) from e

        # This should never be reached due to exceptions, but mypy needs it
        raise RuntimeError("Unexpected code path in create_user")

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID.

        Args:
            user_id: User ID as string

        Returns:
            User dictionary or None if not found

        Raises:
            InvalidUserIdError: If user_id is not numeric
        """
        return await self.get_user_by_id(user_id)

    async def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID.

        Args:
            user_id: Numeric user ID as string

        Returns:
            User dictionary or None if not found

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            # Validate and convert user_id
            try:
                user_id_int = int(user_id)
            except ValueError as e:
                raise InvalidUserIdError(user_id) from e

            result = await self.session.execute(select(User).where(User.id == user_id_int))
            user = result.scalar_one_or_none()

            return self._user_to_dict(user) if user else None

        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise DatabaseOperationError("get", "user", str(e)) from e

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username.

        Args:
            username: Username to search for

        Returns:
            User dictionary or None if not found
        """
        try:
            result = await self.session.execute(select(User).where(User.username == username))
            user = result.scalar_one_or_none()

            return self._user_to_dict(user) if user else None

        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            raise DatabaseOperationError("get", "user", str(e)) from e

        # This should never be reached due to exceptions, but mypy needs it
        raise RuntimeError("Unexpected code path in get_user_by_username")

    async def verify_password(self, username: str, password: str) -> dict[str, Any] | None:
        """Verify user password and return user data if valid.

        Args:
            username: Username to check
            password: Plain text password to verify

        Returns:
            User dictionary if credentials valid, None otherwise
        """
        try:
            user_dict = await self.get_user_by_username(username)
            if not user_dict:
                return None

            if self.pwd_context.verify(password, user_dict["hashed_password"]):
                return user_dict
            return None

        except Exception as e:
            logger.error(f"Failed to verify password for {username}: {e}")
            raise DatabaseOperationError("verify", "password", str(e)) from e

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
        """
        try:
            # Validate and convert user_id
            try:
                user_id_int = int(user_id)
            except ValueError as e:
                raise InvalidUserIdError(user_id) from e

            # Use PostgreSQL's UPDATE ... RETURNING for efficiency
            await self.session.execute(update(User).where(User.id == user_id_int).values(last_login=datetime.now(UTC)))
            await self.session.flush()

            logger.debug(f"Updated last login for user {user_id}")

        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to update last login for user {user_id}: {e}")
            raise DatabaseOperationError("update", "last_login", str(e)) from e

    async def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Get a user by email address.

        Args:
            email: Email address to search for

        Returns:
            User dictionary or None if not found
        """
        try:
            result = await self.session.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()

            return self._user_to_dict(user) if user else None

        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise DatabaseOperationError("get", "user", str(e)) from e

    async def count_users(self, is_active: bool | None = None) -> int:
        """Count total users with optional active filter.

        Args:
            is_active: If provided, only count active/inactive users

        Returns:
            Number of users
        """
        try:
            query = select(func.count(User.id))
            if is_active is not None:
                query = query.where(User.is_active == is_active)

            result = await self.session.scalar(query)
            return result or 0

        except Exception as e:
            logger.error(f"Failed to count users: {e}")
            raise DatabaseOperationError("count", "users", str(e)) from e

    def _user_to_dict(self, user: User | None) -> dict[str, Any] | None:
        """Convert User model to dictionary.

        Args:
            user: User model instance

        Returns:
            User dictionary or None
        """
        if not user:
            return None

        # Helper function to safely convert datetime to string
        def datetime_to_str(dt: Any) -> str | None:
            if dt is None:
                return None
            if hasattr(dt, "isoformat"):
                return dt.isoformat()  # type: ignore[no-any-return]
            # If it's already a string, return it
            return str(dt)

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "hashed_password": user.hashed_password,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
            "created_at": datetime_to_str(user.created_at),
            "updated_at": datetime_to_str(user.updated_at),
            "last_login": datetime_to_str(user.last_login),
        }
