"""Unit tests for reindex handler helpers."""

from __future__ import annotations

import pytest

from webui.tasks.reindex import reindex_handler


class _FakeQdrantManager:
    def __init__(self):
        self.calls = []

    def create_staging_collection(self, base_name: str, vector_size: int) -> str:
        self.calls.append((base_name, vector_size))
        return f"{base_name}-staging"


@pytest.mark.asyncio()
async def test_reindex_handler_requires_vector_store_name() -> None:
    with pytest.raises(ValueError, match="vector_store_name"):
        await reindex_handler({}, {}, _FakeQdrantManager())


@pytest.mark.asyncio()
async def test_reindex_handler_creates_staging_collection() -> None:
    manager = _FakeQdrantManager()
    collection = {"vector_store_name": "base", "config": {"vector_dim": 128}}

    result = await reindex_handler(collection, {"vector_dim": 256}, manager)

    assert result["collection_name"] == "base-staging"
    assert result["vector_dim"] == 256
    assert result["base_collection"] == "base"
    assert manager.calls == [("base", 256)]
