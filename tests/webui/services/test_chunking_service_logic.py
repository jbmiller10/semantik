import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from packages.shared.database.models import OperationStatus
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.dtos.chunking_dtos import ServiceStrategyRecommendation
from packages.webui.services.dtos.api_models import ChunkingStrategy


class FakeResult:
    def __init__(self, *, scalar=None, scalars_list=None, one_dict=None, all_rows=None):
        self._scalar = scalar
        self._scalars_list = scalars_list
        self._one_dict = one_dict
        self._all_rows = all_rows

    def scalar(self):
        return self._scalar

    def one(self):
        return SimpleNamespace(**(self._one_dict or {}))

    def scalars(self):
        data = self._scalars_list or []

        class Proxy:
            def __init__(self, values):
                self._values = values

            def all(self):
                return self._values

        return Proxy(data)

    def all(self):
        return self._all_rows or []

    def __iter__(self):
        return iter(self._all_rows or [])


@pytest.fixture()
def chunking_service(tmp_path):
    db_session = AsyncMock()
    collection_repo = AsyncMock()
    document_repo = AsyncMock()
    service = ChunkingService(db_session, collection_repo, document_repo, redis_client=None)
    service._config_store_path = tmp_path / "chunking_configs.json"
    return service


@pytest.mark.asyncio()
async def test_get_collection_chunks_returns_records(chunking_service):
    chunking_service.collection_repo.get_by_uuid.return_value = SimpleNamespace(id="col-1")

    chunk_obj = SimpleNamespace(
        id=1,
        collection_id="col-1",
        document_id="doc-1",
        chunk_index=0,
        content="Sample chunk",
        token_count=128,
        meta={"source": "unit"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    chunking_service.db_session.execute.side_effect = [
        FakeResult(scalar=1),
        FakeResult(scalars_list=[chunk_obj]),
    ]

    result = await chunking_service.get_collection_chunks("col-1", page=1, page_size=20)

    assert result.total == 1
    assert result.chunks[0].content == "Sample chunk"
    chunking_service.collection_repo.get_by_uuid.assert_called_once()


@pytest.mark.asyncio()
async def test_get_global_metrics_aggregates_counts(chunking_service):
    now = datetime.now(UTC)
    operations = [
        ("col-1", OperationStatus.COMPLETED, now - timedelta(seconds=5), now),
        ("col-2", OperationStatus.FAILED, now - timedelta(seconds=10), now - timedelta(seconds=5)),
    ]

    chunking_service.db_session.execute.side_effect = [
        FakeResult(scalar=2500),
        FakeResult(scalar=125),
        FakeResult(all_rows=operations),
        FakeResult(all_rows=[(ChunkingStrategy.RECURSIVE.value,)]),
    ]

    metrics = await chunking_service.get_global_metrics(period_days=7)

    assert metrics.total_chunks_created == 2500
    assert metrics.total_documents_processed == 125
    assert metrics.most_used_strategy == ChunkingStrategy.RECURSIVE
    assert metrics.total_collections_processed == 1


@pytest.mark.asyncio()
async def test_get_quality_scores_calculates_metrics(chunking_service):
    chunking_service.db_session.execute.side_effect = [
        FakeResult(one_dict={"total_chunks": 100, "avg_length": 200.0, "variance": 100.0}),
        FakeResult(one_dict={"total_docs": 10, "docs_with_chunks": 9}),
    ]

    analysis = await chunking_service.get_quality_scores(collection_id=None)

    assert 0 <= analysis.quality_score <= 1
    assert analysis.recommendations


@pytest.mark.asyncio()
async def test_analyze_document_uses_recommendation(chunking_service):
    chunking_service.recommend_strategy = AsyncMock(
        return_value=ServiceStrategyRecommendation(
            strategy=ChunkingStrategy.RECURSIVE,
            confidence=0.9,
            reasoning="Fits structure",
            chunk_size=1000,
            chunk_overlap=100,
            alternatives=[ChunkingStrategy.MARKDOWN],
        )
    )

    result = await chunking_service.analyze_document(
        content="Paragraph one. Paragraph two.",
        document_id=None,
        file_type="markdown",
        deep_analysis=True,
    )

    assert result.document_type == "markdown"
    assert result.recommended_strategy.strategy == ChunkingStrategy.RECURSIVE
    chunking_service.recommend_strategy.assert_awaited()


@pytest.mark.asyncio()
async def test_save_and_list_configurations(chunking_service):
    chunking_service.config_builder.build_config = lambda strategy, config: chunking_service.config_builder.ChunkingConfigResult(
        strategy=ChunkingStrategy(strategy),
        config=config,
        validation_errors=None,
        warnings=None,
    )

    await chunking_service.save_configuration(
        name="Recursive docs",
        description="Optimised",
        strategy="recursive",
        config={
            "chunk_size": 1200,
            "chunk_overlap": 120,
            "preserve_sentences": True,
        },
        is_default=True,
        tags=["docs"],
        user_id=7,
    )

    configs = await chunking_service.list_configurations(user_id=7)
    assert len(configs) == 1
    assert configs[0].is_default is True
