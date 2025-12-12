from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from webui.tasks import _process_append_operation_impl


class _NullAsyncCM:
    async def __aenter__(self) -> None:  # noqa: D401
        return None

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:  # noqa: ANN401
        return False


class _FakeSession:
    def in_transaction(self) -> bool:
        return False

    def begin_nested(self) -> _NullAsyncCM:
        return _NullAsyncCM()

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


@dataclass(frozen=True)
class _IngestedDoc:
    unique_id: str
    content: str = ""
    content_hash: str = "hash"
    file_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "directory"


class _FakeConnector:
    def __init__(self, docs: list[_IngestedDoc]) -> None:
        self._docs = docs

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self) -> AsyncIterator[_IngestedDoc]:
        for doc in self._docs:
            yield doc


class _FakeRegistry:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ANN401
        return None

    async def register_or_update(self, collection_id: str, ingested: _IngestedDoc, source_id: int) -> dict[str, Any]:
        if ingested.unique_id == "bad-doc":
            raise RuntimeError("registration failed")
        return {
            "is_new": False,
            "is_updated": False,
            "document_id": "ignored",
            "file_size": 1,
        }


class _FakeChunkRepository:
    def __init__(self, session: Any) -> None:  # noqa: ANN401
        self.session = session


@pytest.mark.asyncio()
async def test_append_skips_stale_marking_when_scan_has_errors(monkeypatch) -> None:
    docs = [_IngestedDoc("good-1"), _IngestedDoc("bad-doc"), _IngestedDoc("good-2")]
    connector = _FakeConnector(docs)

    monkeypatch.setattr("webui.tasks.ingestion.ConnectorFactory.get_connector", lambda *_args, **_kwargs: connector)
    monkeypatch.setattr("webui.tasks.ingestion.DocumentRegistryService", _FakeRegistry)
    monkeypatch.setattr("shared.database.repositories.chunk_repository.ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr("webui.tasks.ingestion._audit_log_operation", AsyncMock(return_value=None))

    document_repo = Mock()
    document_repo.session = _FakeSession()
    document_repo.mark_unseen_as_stale = AsyncMock(return_value=999)

    collection_repo = Mock()
    updater = Mock()
    updater.send_update = AsyncMock(return_value=None)

    operation = {
        "id": "op-1",
        "user_id": 1,
        "config": {
            "source_id": 123,
            "source_type": "directory",
            "source_config": {"path": "/tmp"},
        },
    }
    collection = {"id": "col-1"}

    result = await _process_append_operation_impl(operation, collection, collection_repo, document_repo, updater)

    assert result["success"] is False
    assert result["documents_marked_stale"] == 0
    assert document_repo.mark_unseen_as_stale.await_count == 0


@pytest.mark.asyncio()
async def test_append_marks_stale_when_scan_clean(monkeypatch) -> None:
    docs = [_IngestedDoc("good-1"), _IngestedDoc("good-2")]
    connector = _FakeConnector(docs)

    monkeypatch.setattr("webui.tasks.ingestion.ConnectorFactory.get_connector", lambda *_args, **_kwargs: connector)
    monkeypatch.setattr("webui.tasks.ingestion.DocumentRegistryService", _FakeRegistry)
    monkeypatch.setattr("shared.database.repositories.chunk_repository.ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr("webui.tasks.ingestion._audit_log_operation", AsyncMock(return_value=None))

    document_repo = Mock()
    document_repo.session = _FakeSession()
    document_repo.mark_unseen_as_stale = AsyncMock(return_value=2)

    collection_repo = Mock()
    updater = Mock()
    updater.send_update = AsyncMock(return_value=None)

    operation = {
        "id": "op-2",
        "user_id": 1,
        "config": {
            "source_id": 123,
            "source_type": "directory",
            "source_config": {"path": "/tmp"},
        },
    }
    collection = {"id": "col-1"}

    result = await _process_append_operation_impl(operation, collection, collection_repo, document_repo, updater)

    assert result["success"] is True
    assert result["documents_marked_stale"] == 2
    document_repo.mark_unseen_as_stale.assert_awaited_once()
    _call = document_repo.mark_unseen_as_stale.await_args.kwargs
    assert _call["collection_id"] == "col-1"
    assert _call["source_id"] == 123
    assert isinstance(_call["since"], datetime)
