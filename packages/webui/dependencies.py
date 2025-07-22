"""
Common FastAPI dependencies for the WebUI API.
"""

from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import (
    ApiKeyRepository,
    AuthRepository,
    UserRepository,
    create_api_key_repository,
    create_auth_repository,
    create_user_repository,
    get_db,
)
from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from packages.shared.database.models import Collection
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.webui.auth import get_current_user


async def get_collection_for_user(
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Collection:
    """
    FastAPI dependency that retrieves a collection and verifies user ownership.

    Args:
        collection_uuid: UUID of the collection to retrieve
        current_user: Authenticated user dictionary from get_current_user dependency
        db: Database session

    Returns:
        Collection: The Collection ORM object if found and user has access

    Raises:
        HTTPException(404): If collection is not found
        HTTPException(403): If user does not have access to the collection
    """
    repository = CollectionRepository(db)

    try:
        # Convert user_id to int as expected by the repository
        user_id = int(current_user["id"])

        # Use the repository's permission check method
        collection: Collection = await repository.get_by_uuid_with_permission_check(
            collection_uuid=collection_uuid, user_id=user_id
        )
        return collection
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Collection with UUID '{collection_uuid}' not found") from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="You do not have permission to access this collection") from e


async def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    """
    FastAPI dependency that provides a UserRepository instance.

    Args:
        db: Database session from get_db dependency

    Returns:
        UserRepository instance configured with the database session
    """
    return create_user_repository(db)


async def get_auth_repository(db: AsyncSession = Depends(get_db)) -> AuthRepository:
    """
    FastAPI dependency that provides an AuthRepository instance.

    Args:
        db: Database session from get_db dependency

    Returns:
        AuthRepository instance configured with the database session
    """
    return create_auth_repository(db)


async def get_api_key_repository(db: AsyncSession = Depends(get_db)) -> ApiKeyRepository:
    """
    FastAPI dependency that provides an ApiKeyRepository instance.

    Args:
        db: Database session from get_db dependency

    Returns:
        ApiKeyRepository instance configured with the database session
    """
    return create_api_key_repository(db)
