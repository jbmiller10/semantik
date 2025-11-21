import pytest
from fastapi import HTTPException

from webui.api.v2 import chunking as chunking_module


class _Req:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload


class _CollectionService:
    async def create_operation(self, **_kwargs):
        return {"uuid": "op-123"}

    async def update_collection(self, **_kwargs):
        return None


class _Validator:
    def validate_strategy(self, *_args, **_kwargs):
        return None

    def validate_config(self, *_args, **_kwargs):
        return None


class _Service:
    def __init__(self) -> None:
        self.validator = _Validator()


@pytest.mark.asyncio()
async def test_start_chunking_operation_enqueues(monkeypatch):
    sent_task = {}

    class _CeleryApp:
        def send_task(self, name, args=None, kwargs=None):  # noqa: D401
            sent_task["task"] = (name, args, kwargs)
            return "ok"

    monkeypatch.setattr("webui.tasks.celery_app", _CeleryApp())

    req = _Req(
        {
            "strategy": "recursive",
            "config": {"strategy": "recursive", "chunk_size": 1000},
            "document_ids": [],
            "priority": "normal",
        }
    )

    result = await chunking_module.start_chunking_operation(
        request=req,
        collection_uuid="col-1",
        _current_user={"id": 1},
        collection={},
        service=_Service(),
        collection_service=_CollectionService(),
    )

    assert result.operation_id == "op-123"
    assert sent_task["task"][0] == "webui.tasks.process_collection_operation"


@pytest.mark.asyncio()
async def test_update_chunking_strategy_returns_completed_without_reprocess():
    calls = {"updated": False}

    class _CollectionServiceCompleted(_CollectionService):
        async def update_collection(self, **_kwargs):  # noqa: D401
            calls["updated"] = True
            # Implicit None return

    req = _Req(
        {
            "strategy": "recursive",
            "config": {"strategy": "recursive", "chunk_size": 1000},
            "reprocess_existing": False,
        }
    )

    result = await chunking_module.update_chunking_strategy(
        collection_id="col-1",
        request=req,
        _current_user={"id": 1},
        collection={},
        service=_Service(),
        collection_service=_CollectionServiceCompleted(),
    )

    assert calls["updated"] is True
    assert result.status.value == "completed"


@pytest.mark.asyncio()
async def test_update_chunking_strategy_reprocess_enqueues(monkeypatch):
    sent = {}

    class _CeleryApp:
        def send_task(self, *args, **kwargs):  # noqa: D401
            sent["call"] = (args, kwargs)
            return "ok"

    monkeypatch.setattr("webui.tasks.celery_app", _CeleryApp())

    req = _Req(
        {
            "strategy": "recursive",
            "config": {"strategy": "recursive", "chunk_size": 1000},
            "reprocess_existing": True,
        }
    )

    result = await chunking_module.update_chunking_strategy(
        collection_id="col-1",
        request=req,
        _current_user={"id": 1},
        collection={},
        service=_Service(),
        collection_service=_CollectionService(),
    )

    assert result.status.value == "pending"
    assert sent["call"][0][0] == "webui.tasks.process_collection_operation"


@pytest.mark.asyncio()
async def test_start_chunking_operation_raises_when_celery_unavailable(monkeypatch):
    called = {}

    class _CeleryApp:
        def send_task(self, *_args, **_kwargs):
            called["sent"] = True
            raise RuntimeError("no broker")

    monkeypatch.setattr("webui.tasks.celery_app", _CeleryApp())

    req = _Req(
        {
            "strategy": "recursive",
            "config": {"strategy": "recursive", "chunk_size": 1000},
            "document_ids": [],
            "priority": "normal",
        }
    )

    with pytest.raises(HTTPException) as excinfo:
        await chunking_module.start_chunking_operation(
            request=req,
            collection_uuid="col-1",
            _current_user={"id": 1},
            collection={},
            service=_Service(),
            collection_service=_CollectionService(),
        )

    assert excinfo.value.status_code == 503
    assert "could not be queued" in excinfo.value.detail
    assert called.get("sent")


@pytest.mark.asyncio()
async def test_update_chunking_strategy_fails_when_reprocess_enqueue_fails(monkeypatch):
    class _CeleryApp:
        def send_task(self, *_args, **_kwargs):
            raise RuntimeError("no broker")

    monkeypatch.setattr("webui.tasks.celery_app", _CeleryApp())

    req = _Req(
        {
            "strategy": "recursive",
            "config": {"strategy": "recursive", "chunk_size": 1000},
            "reprocess_existing": True,
        }
    )

    with pytest.raises(HTTPException):
        await chunking_module.update_chunking_strategy(
            collection_id="col-1",
            request=req,
            _current_user={"id": 1},
            collection={},
            service=_Service(),
            collection_service=_CollectionService(),
        )
