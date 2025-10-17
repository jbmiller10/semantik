"""Shared fixtures for v2 API integration tests."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

from packages.shared.database import get_db
from packages.webui.auth import create_access_token, get_current_user
from packages.webui.dependencies import get_collection_for_user as original_get_collection_for_user
from packages.webui.main import app
from packages.webui.services import factory as services_factory


@pytest.fixture()
def _reset_redis_manager() -> None:
    """Ensure Redis manager singleton does not leak between tests."""

    services_factory._redis_manager = None
    yield
    services_factory._redis_manager = None


@pytest.fixture()
def api_auth_headers(test_user_db) -> dict[str, str]:
    """Issue a valid bearer token for the persisted test user."""

    token = create_access_token(data={"sub": test_user_db.username})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture()
async def api_client(
    db_session,
    test_user_db,
    _use_fakeredis,
    _reset_redis_manager,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an AsyncClient with real DB session and fakeredis overrides."""

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    async def override_get_current_user() -> dict[str, Any]:
        return {
            "id": test_user_db.id,
            "username": test_user_db.username,
            "email": test_user_db.email,
            "full_name": test_user_db.full_name,
        }

    async def override_get_collection_for_user(
        collection_uuid: str,
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> Any:
        return await original_get_collection_for_user(
            collection_uuid=collection_uuid,
            current_user=current_user,
            db=db_session,
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[original_get_collection_for_user] = override_get_collection_for_user

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
