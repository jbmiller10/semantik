"""Exercise the append-task compatibility wrapper with stubbed dependencies."""

import pytest

from webui.tasks import ingestion as ingestion_module


class _FakeResult:
    def __init__(self, value=None):
        self._value = value

    def scalar_one(self):
        return self._value

    def scalar_one_or_none(self):  # noqa: D401
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._value


class _FakeSession:
    def __init__(self, op, collection, docs):
        self._results = [op, collection, docs]

    async def execute(self, _stmt):
        return _FakeResult(self._results.pop(0))


class _Doc:
    def __init__(self):
        from shared.database.models import DocumentStatus

        self.id = "doc-1"
        self.file_path = "/tmp/doc1.txt"
        self.chunk_count = 0
        self.status = DocumentStatus.PENDING


class _Op:
    def __init__(self):
        self.id = "op-1"
        self.uuid = "op-1"
        self.collection_id = "col-1"


class _Collection:
    def __init__(self):
        self.id = "col-1"
        self.name = "Demo"
        self.chunking_strategy = "recursive"
        self.chunking_config = {"chunk_size": 1000}
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.embedding_model = "model"
        self.quantization = "float16"
        self.vector_store_name = "vs-1"


class _Updater:
    def __init__(self):
        self.updates = []

    async def send_update(self, kind, payload):  # noqa: D401
        self.updates.append((kind, payload))


class _ChunkingService:
    async def execute_ingestion_chunking(self, **_kwargs):  # noqa: D401
        return [{"text": "chunk", "metadata": _kwargs.get("metadata", {})}]


@pytest.mark.asyncio()
async def test_process_append_operation_uses_orchestrator(monkeypatch):
    monkeypatch.setattr(
        ingestion_module,
        "extract_and_serialize_thread_safe",
        lambda _path: [("hello", {"a": 1})],
    )

    monkeypatch.setattr(
        "webui.tasks.celery_app",
        type(
            "_Celery",
            (),
            {"send_task": lambda *args, **kwargs: None},
        ),
    )

    async def fake_resolver(*_args, **_kwargs):  # noqa: D401
        return _ChunkingService()

    tasks_ns = ingestion_module._tasks_namespace()
    monkeypatch.setattr(tasks_ns, "resolve_celery_chunking_orchestrator", fake_resolver, raising=False)
    monkeypatch.setattr(tasks_ns, "extract_and_serialize_thread_safe", lambda path: [("hello", {})], raising=False)

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            self.posts = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, json=None, headers=None):  # noqa: D401
            self.posts.append((url, json))
            return None

    monkeypatch.setattr(ingestion_module.httpx, "AsyncClient", _AsyncClient)

    db = _FakeSession(_Op(), _Collection(), [_Doc()])
    updater = _Updater()

    result = await ingestion_module._process_append_operation(db=db, updater=updater, _operation_id="op-1")

    assert result["success"] is True
    assert updater.updates, "Progress updates should be sent"
