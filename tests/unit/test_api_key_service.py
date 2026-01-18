from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest

from shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, EntityNotFoundError, ValidationError
from shared.database.models import ApiKey
from webui.api.v2.api_key_schemas import ApiKeyCreate
from webui.services.api_key_service import ApiKeyService


class _ScalarResult:
    def __init__(self, *, scalar_one=None, scalar_one_or_none=None, scalars_all=None):
        self._scalar_one = scalar_one
        self._scalar_one_or_none = scalar_one_or_none
        self._scalars_all = scalars_all

    def scalar_one(self):
        return self._scalar_one

    def scalar_one_or_none(self):
        return self._scalar_one_or_none

    def scalars(self):
        return self

    def all(self):
        return self._scalars_all


def test_generate_key_format_includes_uuid_prefix() -> None:
    key_id = "550e8400-e29b-41d4-a716-446655440000"
    raw_key = ApiKeyService._generate_key(key_id)

    assert raw_key.startswith("smtk_550e8400_")
    parts = raw_key.split("_", 2)
    assert parts[0] == "smtk"
    assert parts[1] == "550e8400"
    assert len(parts[2]) >= 20


def test_hash_key_is_sha256_hex() -> None:
    raw_key = "smtk_550e8400_secret"
    hashed1 = ApiKeyService._hash_key(raw_key)
    hashed2 = ApiKeyService._hash_key(raw_key)

    assert hashed1 == hashed2
    assert len(hashed1) == 64
    assert all(c in "0123456789abcdef" for c in hashed1)


@pytest.mark.asyncio()
async def test_count_user_active_keys_executes_query() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalar_one=3))
    service = ApiKeyService(session)

    assert await service._count_user_active_keys(123) == 3


@pytest.mark.asyncio()
async def test_get_by_name_executes_query() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalar_one_or_none=None))
    service = ApiKeyService(session)

    assert await service._get_by_name("Example", 123) is None


@pytest.mark.asyncio()
async def test_create_raises_when_key_limit_reached() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalar_one=2))
    service = ApiKeyService(session)

    with patch(
        "webui.services.api_key_service.settings",
        new=SimpleNamespace(API_KEY_MAX_PER_USER=2, API_KEY_DEFAULT_EXPIRY_DAYS=30, API_KEY_MAX_EXPIRY_DAYS=0),
    ):
        with pytest.raises(ValidationError, match="Maximum number of active API keys"):
            await service.create(ApiKeyCreate(name="test", expires_in_days=30), user_id=123)


@pytest.mark.asyncio()
async def test_create_raises_when_name_exists_case_insensitive() -> None:
    existing = ApiKey(id="k1", user_id=123, name="Existing", key_hash="h1", permissions=None)

    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            _ScalarResult(scalar_one=0),
            _ScalarResult(scalar_one_or_none=existing),
        ]
    )
    service = ApiKeyService(session)

    with patch(
        "webui.services.api_key_service.settings",
        new=SimpleNamespace(API_KEY_MAX_PER_USER=2, API_KEY_DEFAULT_EXPIRY_DAYS=30, API_KEY_MAX_EXPIRY_DAYS=0),
    ):
        with pytest.raises(EntityAlreadyExistsError):
            await service.create(ApiKeyCreate(name="existing", expires_in_days=30), user_id=123)


@pytest.mark.asyncio()
async def test_create_clamps_max_expiry_and_persists() -> None:
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            _ScalarResult(scalar_one=0),
            _ScalarResult(scalar_one_or_none=None),
        ]
    )
    service = ApiKeyService(session)

    fixed_uuid = UUID("550e8400-e29b-41d4-a716-446655440000")
    before = datetime.now(UTC)

    with (
        patch(
            "webui.services.api_key_service.settings",
            new=SimpleNamespace(API_KEY_MAX_PER_USER=2, API_KEY_DEFAULT_EXPIRY_DAYS=30, API_KEY_MAX_EXPIRY_DAYS=10),
        ),
        patch("webui.services.api_key_service.uuid4", return_value=fixed_uuid),
        patch("webui.services.api_key_service.secrets.token_urlsafe", return_value="secret"),
    ):
        api_key, raw_key = await service.create(ApiKeyCreate(name="My Key", expires_in_days=365), user_id=123)

    after = datetime.now(UTC)

    assert api_key.id == str(fixed_uuid)
    assert api_key.user_id == 123
    assert api_key.name == "My Key"
    assert api_key.is_active is True
    assert raw_key.startswith("smtk_550e8400_")
    assert api_key.key_hash == ApiKeyService._hash_key(raw_key)
    assert before + timedelta(days=10) <= api_key.expires_at <= after + timedelta(days=10)
    session.add.assert_called_once()
    session.flush.assert_awaited_once()


@pytest.mark.asyncio()
async def test_create_uses_default_expiry_when_not_provided() -> None:
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            _ScalarResult(scalar_one=0),
            _ScalarResult(scalar_one_or_none=None),
        ]
    )
    service = ApiKeyService(session)

    fixed_uuid = UUID("550e8400-e29b-41d4-a716-446655440000")
    before = datetime.now(UTC)

    with (
        patch(
            "webui.services.api_key_service.settings",
            new=SimpleNamespace(API_KEY_MAX_PER_USER=2, API_KEY_DEFAULT_EXPIRY_DAYS=7, API_KEY_MAX_EXPIRY_DAYS=0),
        ),
        patch("webui.services.api_key_service.uuid4", return_value=fixed_uuid),
        patch("webui.services.api_key_service.secrets.token_urlsafe", return_value="secret"),
    ):
        api_key, _ = await service.create(ApiKeyCreate(name="My Key", expires_in_days=None), user_id=123)

    after = datetime.now(UTC)
    assert before + timedelta(days=7) <= api_key.expires_at <= after + timedelta(days=7)


@pytest.mark.asyncio()
async def test_list_for_user_returns_keys() -> None:
    key1 = ApiKey(id="k1", user_id=123, name="Key 1", key_hash="h1", permissions=None)
    key2 = ApiKey(id="k2", user_id=123, name="Key 2", key_hash="h2", permissions=None)

    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalars_all=[key1, key2]))
    service = ApiKeyService(session)

    keys = await service.list_for_user(123)
    assert [key.id for key in keys] == ["k1", "k2"]


@pytest.mark.asyncio()
async def test_get_raises_for_missing_key() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalar_one_or_none=None))
    service = ApiKeyService(session)

    with pytest.raises(EntityNotFoundError):
        await service.get("missing", user_id=123)


@pytest.mark.asyncio()
async def test_get_raises_for_wrong_owner() -> None:
    key = ApiKey(id="k1", user_id=1, name="Key", key_hash="h", permissions=None)
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalar_one_or_none=key))
    service = ApiKeyService(session)

    with pytest.raises(AccessDeniedError):
        await service.get("k1", user_id=2)


@pytest.mark.asyncio()
async def test_get_returns_key_for_owner() -> None:
    key = ApiKey(id="k1", user_id=123, name="Key", key_hash="h", permissions=None)
    session = AsyncMock()
    session.execute = AsyncMock(return_value=_ScalarResult(scalar_one_or_none=key))
    service = ApiKeyService(session)

    result = await service.get("k1", user_id=123)
    assert result is key


@pytest.mark.asyncio()
async def test_update_active_status_reactivation_enforces_max_active_keys() -> None:
    key = ApiKey(id="k1", user_id=123, name="Key", key_hash="h", permissions=None, is_active=False)
    session = AsyncMock()
    session.flush = AsyncMock()
    service = ApiKeyService(session)

    with (
        patch("webui.services.api_key_service.settings", new=SimpleNamespace(API_KEY_MAX_PER_USER=2)),
        patch.object(service, "get", AsyncMock(return_value=key)),
        patch.object(service, "_count_user_active_keys", AsyncMock(return_value=2)),
    ):
        with pytest.raises(ValidationError, match="Maximum number of active API keys"):
            await service.update_active_status("k1", user_id=123, is_active=True)


@pytest.mark.asyncio()
async def test_update_active_status_revokes_key() -> None:
    key = ApiKey(id="k1", user_id=123, name="Key", key_hash="h", permissions=None, is_active=True)
    session = AsyncMock()
    session.flush = AsyncMock()
    service = ApiKeyService(session)

    with patch.object(service, "get", AsyncMock(return_value=key)):
        updated = await service.update_active_status("k1", user_id=123, is_active=False)

    assert updated.is_active is False
    session.flush.assert_awaited_once()
