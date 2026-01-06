from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from shared.database.repositories.chunk_repository import ChunkRepository


@pytest.mark.asyncio()
async def test_get_chunk_by_metadata_chunk_id_validates_input() -> None:
    repo = ChunkRepository(MagicMock())
    with pytest.raises(ValueError, match="non-empty"):
        await repo.get_chunk_by_metadata_chunk_id("", str(uuid4()))


@pytest.mark.asyncio()
async def test_get_chunk_by_metadata_chunk_id_executes_query() -> None:
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)

    repo = ChunkRepository(session)
    chunk = await repo.get_chunk_by_metadata_chunk_id("doc_0001", str(uuid4()))
    assert chunk is None
    session.execute.assert_awaited()
