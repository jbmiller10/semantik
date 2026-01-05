"""Unit tests for ingestion helpers."""

from __future__ import annotations

from webui.tasks.ingestion import _get_embedding_concurrency


def test_get_embedding_concurrency_default(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_CONCURRENCY_PER_WORKER", raising=False)
    assert _get_embedding_concurrency() == 1


def test_get_embedding_concurrency_valid(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "4")
    assert _get_embedding_concurrency() == 4


def test_get_embedding_concurrency_minimum(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "0")
    assert _get_embedding_concurrency() == 1


def test_get_embedding_concurrency_invalid(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "not-a-number")
    assert _get_embedding_concurrency() == 1
