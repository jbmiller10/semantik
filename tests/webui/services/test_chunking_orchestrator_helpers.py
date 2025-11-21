"""Unit coverage for orchestrator helper methods (pure logic paths)."""

import pytest

from shared.chunking.infrastructure.exceptions import ValidationError
from webui.services.chunking import ChunkingCache, ChunkingConfigManager, ChunkingProcessor, ChunkingValidator
from webui.services.chunking.orchestrator import ChunkingOrchestrator
from webui.services.dtos import ServiceChunkPreview, ServiceStrategyComparison, ServiceStrategyMetrics


class _DummyMetrics:
    def __init__(self) -> None:
        self.called_with: int | None = None

    async def get_metrics_by_strategy(self, period_days: int = 30):  # noqa: D401, ARG002
        self.called_with = period_days
        return ServiceStrategyMetrics.create_default_metrics()

    def get_strategy_metrics(self, strategy: str) -> dict:
        return {"average_duration": 1.0, "success_rate": 0.9, "strategy": strategy}


def _orchestrator(metrics: object = None) -> ChunkingOrchestrator:
    return ChunkingOrchestrator(
        processor=ChunkingProcessor(),
        cache=ChunkingCache(redis_client=None),
        metrics=metrics or _DummyMetrics(),
        validator=ChunkingValidator(),
        config_manager=ChunkingConfigManager(profile_repo=None),
    )


@pytest.mark.asyncio()
async def test_get_metrics_by_strategy_passes_params():
    metrics = _DummyMetrics()
    orch = _orchestrator(metrics=metrics)

    result = await orch.get_metrics_by_strategy(period_days=7)

    assert metrics.called_with == 7
    assert result
    assert isinstance(result[0], ServiceStrategyMetrics)


@pytest.mark.asyncio()
async def test_get_quality_scores_uses_collection_stats(monkeypatch):
    orch = _orchestrator()

    class _Stats:
        avg_chunk_size = 150

    called: dict[str, object] = {}

    async def fake_stats(collection_id: str, user_id: int):  # noqa: D401
        called["collection_id"] = collection_id
        called["user_id"] = user_id
        return _Stats()

    monkeypatch.setattr(orch, "get_collection_statistics", fake_stats)

    scores = await orch.get_quality_scores(collection_id="abc", user_id=42)

    assert scores.issues_detected, "Small chunk size should register an issue"
    assert scores.quality_score < 0.8
    assert called == {"collection_id": "abc", "user_id": 42}


@pytest.mark.asyncio()
async def test_get_quality_scores_requires_user_for_collection():
    orch = _orchestrator()

    with pytest.raises(ValidationError):
        await orch.get_quality_scores(collection_id="abc", user_id=None)


@pytest.mark.asyncio()
async def test_analyze_document_returns_recommendation():
    orch = _orchestrator()
    analysis = await orch.analyze_document(content="# Title\nhello" * 50, file_type="md")

    assert analysis.document_type in {"md", "markdown"}
    assert analysis.estimated_chunks["recursive"] >= 1


def test_normalize_and_preview_helpers_cover_transformations():
    orch = _orchestrator()

    normalized = orch._normalize_config({"params": {"chunk_size": 123}, "other": True})
    assert normalized["chunk_size"] == 123
    assert normalized["other"] is True

    chunks = orch._transform_chunks_to_preview(
        [
            {"content": "hello", "token_count": 3, "metadata": {"k": "v"}},
            ServiceChunkPreview(index=1, content="hi", token_count=1, char_count=2, metadata={}),
        ]
    )
    metrics = orch._calculate_preview_metrics(chunks, text_length=20, processing_time=0.5)
    assert metrics["total_chunks"] == 2

    cached = orch._build_preview_response_from_cache(
        {
            "preview_id": "cache-1",
            "chunks": [chunk.__dict__ for chunk in chunks],
            "metrics": {"m": 1},
            "config": {"x": 1},
            "strategy": "recursive",
            "total_chunks": 2,
            "processing_time_ms": 5,
            "expires_at": "2099-01-01T00:00:00",
        },
        correlation_id="corr-1",
        max_chunks=1,
    )
    assert cached.cached is True
    assert cached.total_chunks == 2


def test_strategy_quality_helpers():
    orch = _orchestrator()
    stats = {"avg_chunk_size": 600, "min_chunk_size": 500, "max_chunk_size": 700}
    quality = orch._calculate_quality_score(stats)
    variance = orch._calculate_variance(stats)

    comp = [
        ServiceStrategyComparison(
            strategy="recursive",
            config={"chunk_size": 600},
            sample_chunks=[],
            total_chunks=2,
            avg_chunk_size=600,
            size_variance=variance,
            quality_score=quality,
            processing_time_ms=10,
        )
    ]
    recommendation = orch._get_recommendation(comp, content="text")

    assert 0 <= quality <= 1
    assert variance >= 0
    assert recommendation.strategy == "recursive"
