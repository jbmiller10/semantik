from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from webui.api.v2.api_key_schemas import ApiKeyCreate, ApiKeyUpdate
from webui.api.v2.api_keys import _api_key_to_response, create_api_key, get_api_key, list_api_keys, update_api_key


def _make_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )


@dataclass
class _ApiKey:
    id: str
    name: str
    is_active: bool
    permissions: dict[str, Any] | None
    last_used_at: datetime | None
    expires_at: datetime | None
    created_at: datetime


def test_api_key_to_response_maps_fields() -> None:
    now = datetime.now(UTC)
    api_key = _ApiKey(
        id="k1",
        name="Key",
        is_active=True,
        permissions=None,
        last_used_at=None,
        expires_at=now + timedelta(days=1),
        created_at=now,
    )

    response = _api_key_to_response(api_key)
    assert response.id == "k1"
    assert response.name == "Key"
    assert response.is_active is True
    assert response.expires_at == api_key.expires_at


@pytest.mark.asyncio()
async def test_api_key_endpoints_happy_path() -> None:
    now = datetime.now(UTC)
    api_key = _ApiKey(
        id="k1",
        name="Key",
        is_active=True,
        permissions=None,
        last_used_at=None,
        expires_at=now + timedelta(days=1),
        created_at=now,
    )

    service = AsyncMock()
    service.create.return_value = (api_key, "smtk_12345678_" + ("x" * 32))
    service.list_for_user.return_value = [api_key]
    service.get.return_value = api_key
    service.update_active_status.return_value = api_key

    current_user = {"id": 123}
    request = _make_request()

    created = await create_api_key.__wrapped__(
        request,
        ApiKeyCreate(name="Key", expires_in_days=30),
        current_user,
        service,
    )
    assert created.id == "k1"
    assert created.api_key.startswith("smtk_")

    listed = await list_api_keys.__wrapped__(request, current_user, service)
    assert listed.total == 1
    assert listed.api_keys[0].id == "k1"

    fetched = await get_api_key.__wrapped__(request, "k1", current_user, service)
    assert fetched.id == "k1"

    updated = await update_api_key.__wrapped__(request, "k1", ApiKeyUpdate(is_active=True), current_user, service)
    assert updated.id == "k1"
