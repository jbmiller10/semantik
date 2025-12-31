import logging
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


class _FailingRollbackSession(_FakeSession):
    def __init__(self) -> None:
        self._commit_calls = 0

    def in_transaction(self) -> bool:
        return self._commit_calls == 0

    async def commit(self) -> None:
        self._commit_calls += 1
        if self._commit_calls == 1:
            raise RuntimeError("commit failed")

    async def rollback(self) -> None:
        raise RuntimeError("rollback failed")


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


class _FakeCredentialedConnector(_FakeConnector):
    def __init__(self, docs: list[_IngestedDoc]) -> None:
        super().__init__(docs)
        self.token: str | None = None

    def set_credentials(self, token: str | None = None) -> None:
        self.token = token

    async def authenticate(self) -> bool:
        return self.token == "secret-token"


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


@pytest.mark.asyncio()
async def test_append_logs_rollback_failure(monkeypatch, caplog) -> None:
    docs: list[_IngestedDoc] = []
    connector = _FakeConnector(docs)

    monkeypatch.setattr("webui.tasks.ingestion.ConnectorFactory.get_connector", lambda *_args, **_kwargs: connector)
    monkeypatch.setattr("webui.tasks.ingestion.DocumentRegistryService", _FakeRegistry)
    monkeypatch.setattr("shared.database.repositories.chunk_repository.ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr("webui.tasks.ingestion._audit_log_operation", AsyncMock(return_value=None))

    document_repo = Mock()
    document_repo.session = _FailingRollbackSession()
    document_repo.mark_unseen_as_stale = AsyncMock(return_value=0)

    collection_repo = Mock()
    updater = Mock()
    updater.send_update = AsyncMock(return_value=None)

    operation = {
        "id": "op-rollback",
        "user_id": 1,
        "config": {
            "source_id": 456,
            "source_type": "directory",
            "source_config": {"path": "/tmp"},
        },
    }
    collection = {"id": "col-rollback"}

    with caplog.at_level(logging.WARNING):
        await _process_append_operation_impl(operation, collection, collection_repo, document_repo, updater)

    assert "Failed to rollback after pre-scan commit error" in caplog.text
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


@pytest.mark.asyncio()
async def test_append_applies_stored_secrets_before_authenticate(monkeypatch) -> None:
    class _FakeSecretRepo:
        def __init__(self, session: Any) -> None:  # noqa: ANN401
            self.session = session

        async def get_secret_types_for_source(self, source_id: int) -> list[str]:  # noqa: ARG002
            # Include an extra secret type that the connector does NOT accept to
            # verify ingestion filters kwargs against set_credentials signature.
            return ["token", "ssh_key"]

        async def get_secret(self, source_id: int, secret_type: str) -> str | None:  # noqa: ARG002
            return {"token": "secret-token", "ssh_key": "ssh-private-key"}.get(secret_type)

    docs: list[_IngestedDoc] = []
    connector = _FakeCredentialedConnector(docs)

    monkeypatch.setattr("webui.tasks.ingestion.ConnectorFactory.get_connector", lambda *_args, **_kwargs: connector)
    monkeypatch.setattr("webui.tasks.ingestion.DocumentRegistryService", _FakeRegistry)
    monkeypatch.setattr("shared.database.repositories.chunk_repository.ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr("webui.tasks.ingestion._audit_log_operation", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "shared.database.repositories.connector_secret_repository.ConnectorSecretRepository",
        _FakeSecretRepo,
    )

    document_repo = Mock()
    document_repo.session = _FakeSession()
    document_repo.mark_unseen_as_stale = AsyncMock(return_value=0)

    collection_repo = Mock()
    updater = Mock()
    updater.send_update = AsyncMock(return_value=None)

    operation = {
        "id": "op-secrets",
        "user_id": 1,
        "config": {
            "source_id": 123,
            "source_type": "git",
            "source_config": {"url": "https://example.com/repo.git"},
        },
    }
    collection = {"id": "col-1"}

    result = await _process_append_operation_impl(operation, collection, collection_repo, document_repo, updater)

    assert result["success"] is True
    assert connector.token == "secret-token"


@pytest.mark.asyncio()
async def test_append_skips_stale_marking_when_processing_has_failures(monkeypatch) -> None:
    class _FakeRegistryNewDoc(_FakeRegistry):
        async def register_or_update(
            self,
            collection_id: str,
            ingested: _IngestedDoc,
            source_id: int,
        ) -> dict[str, Any]:
            return {
                "is_new": True,
                "is_updated": False,
                "document_id": 101,
                "file_size": 1,
            }

    class _FakeQdrantClient:
        def get_collection(self, _name: str) -> None:
            return None

    class _FakeQdrantManager:
        def get_client(self) -> _FakeQdrantClient:
            return _FakeQdrantClient()

    class _FailingChunkingService:
        async def execute_ingestion_chunking(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
            raise RuntimeError("chunking failed")

    async def _fake_resolve_chunker(*_args: Any, **_kwargs: Any) -> _FailingChunkingService:  # noqa: ANN401
        return _FailingChunkingService()

    docs = [_IngestedDoc("good-1", content="hello")]
    connector = _FakeConnector(docs)

    monkeypatch.setattr("webui.tasks.ingestion.ConnectorFactory.get_connector", lambda *_args, **_kwargs: connector)
    monkeypatch.setattr("webui.tasks.ingestion.DocumentRegistryService", _FakeRegistryNewDoc)
    monkeypatch.setattr("shared.database.repositories.chunk_repository.ChunkRepository", _FakeChunkRepository)
    monkeypatch.setattr("webui.tasks.ingestion._audit_log_operation", AsyncMock(return_value=None))
    monkeypatch.setattr("webui.tasks.ingestion.resolve_qdrant_manager", lambda: _FakeQdrantManager())
    monkeypatch.setattr("webui.tasks.ingestion.resolve_celery_chunking_orchestrator", _fake_resolve_chunker)

    document_repo = Mock()
    document_repo.session = _FakeSession()
    document_repo.mark_unseen_as_stale = AsyncMock(return_value=999)

    fake_doc = Mock()
    fake_doc.id = 101
    fake_doc.file_path = "/tmp/x.txt"
    fake_doc.uri = None
    fake_doc.chunk_count = 0

    document_repo.get_by_id = AsyncMock(return_value=fake_doc)
    document_repo.update_status = AsyncMock(return_value=None)
    document_repo.get_stats_by_collection = AsyncMock(return_value={"total_documents": 1})

    collection_repo = Mock()
    collection_repo.update_stats = AsyncMock(return_value=None)

    updater = Mock()
    updater.send_update = AsyncMock(return_value=None)

    operation = {
        "id": "op-3",
        "user_id": 1,
        "config": {
            "source_id": 123,
            "source_type": "directory",
            "source_config": {"path": "/tmp"},
            "batch_size": 1,
        },
    }
    collection = {
        "id": "col-1",
        "vector_store_name": "qdrant-col-1",
    }

    result = await _process_append_operation_impl(operation, collection, collection_repo, document_repo, updater)

    assert result["success"] is False
    assert result["documents_marked_stale"] == 0
    assert document_repo.mark_unseen_as_stale.await_count == 0
