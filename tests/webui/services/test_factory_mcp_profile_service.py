from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from webui.services.factory import create_mcp_profile_service, get_mcp_profile_service


def test_create_mcp_profile_service_constructs_service() -> None:
    db = MagicMock()
    service = create_mcp_profile_service(db)
    assert service.db_session is db


@pytest.mark.asyncio()
async def test_get_mcp_profile_service_returns_service() -> None:
    db = MagicMock()
    service = await get_mcp_profile_service(db=db)
    assert service.db_session is db
