"""
Common FastAPI dependencies for the WebUI API.
"""

from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import get_db
from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityNotFoundError,
)
from packages.shared.database.models import Collection
from packages.shared.database.repositories.collection_repository import (
    CollectionRepository,
)
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
