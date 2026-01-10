from __future__ import annotations

from webui import sparse_tasks


def test_reindex_sparse_collection_defaults_model_config(monkeypatch) -> None:
    captured = {}

    async def _fake_reindex(task, collection_uuid, plugin_id, model_config):  # type: ignore[no-untyped-def]
        captured["model_config"] = model_config
        return {"status": "completed"}

    monkeypatch.setattr(sparse_tasks, "_reindex_collection_async", _fake_reindex)

    # Call the task run method directly to avoid passing task self explicitly
    result = sparse_tasks.reindex_sparse_collection.run("col-1", "bm25-local", None)

    assert result["status"] == "completed"
    assert captured["model_config"] == {}
