"""Shared fixtures for v2 API integration tests."""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

import pytest
import pytest_asyncio
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

from shared.database import get_db
from shared.database.models import LLMProviderConfig
from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.utils.encryption import SecretEncryption, generate_fernet_key
from webui.auth import create_access_token, get_current_user
from webui.dependencies import get_collection_for_user as original_get_collection_for_user
from webui.main import app
from webui.services import factory as services_factory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


@pytest.fixture(autouse=True)
def _initialize_encryption():
    """Initialize encryption for all LLM API tests."""
    test_key = generate_fernet_key()
    SecretEncryption.initialize(test_key)
    yield
    SecretEncryption.reset()


@pytest_asyncio.fixture()
async def llm_config(db_session: "AsyncSession", test_user_db: "User") -> LLMProviderConfig:
    """Create a basic LLMProviderConfig for testing."""
    config = LLMProviderConfig(
        user_id=test_user_db.id,
        high_quality_provider="anthropic",
        high_quality_model="claude-opus-4-5-20251101",
        low_quality_provider="anthropic",
        low_quality_model="claude-sonnet-4-5-20250929",
    )
    db_session.add(config)
    await db_session.flush()
    return config


@pytest_asyncio.fixture()
async def llm_config_with_key(db_session: "AsyncSession", llm_config: LLMProviderConfig) -> LLMProviderConfig:
    """LLMProviderConfig with an encrypted API key."""
    repo = LLMProviderConfigRepository(db_session)
    config_id = cast(int, llm_config.id)
    await repo.set_api_key(config_id, "anthropic", "sk-ant-test-key-for-integration-tests")
    await db_session.flush()
    return llm_config


@pytest.fixture(name="reset_redis_manager")
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
    use_fakeredis,
    reset_redis_manager,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an AsyncClient with real DB session and fakeredis overrides."""

    _ = use_fakeredis
    _ = reset_redis_manager

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


@pytest_asyncio.fixture()
async def api_client_unauthenticated(
    db_session,
    use_fakeredis,
    reset_redis_manager,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an AsyncClient WITHOUT auth override for testing auth requirements."""

    _ = use_fakeredis
    _ = reset_redis_manager

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    # Note: get_current_user is NOT overridden - authentication is enforced

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def other_user_auth_headers(other_user_db) -> dict[str, str]:
    """Issue a valid bearer token for the other test user."""

    token = create_access_token(data={"sub": other_user_db.username})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture()
async def api_client_other_user(
    db_session,
    other_user_db,
    use_fakeredis,
    reset_redis_manager,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an AsyncClient authenticated as the other test user."""

    _ = use_fakeredis
    _ = reset_redis_manager

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    async def override_get_current_user() -> dict[str, Any]:
        return {
            "id": other_user_db.id,
            "username": other_user_db.username,
            "email": other_user_db.email,
            "full_name": other_user_db.full_name,
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
