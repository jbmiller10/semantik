from datetime import UTC, datetime

from packages.webui.services.dtos.api_models import ChunkingStrategy
from packages.webui.services.dtos.chunking_dtos import (
    ServiceChunkList,
    ServiceChunkRecord,
    ServiceDocumentAnalysis,
    ServiceGlobalMetrics,
    ServiceQualityAnalysis,
    ServiceSavedConfiguration,
    ServiceStrategyRecommendation,
)


def test_service_chunk_record_to_api_dict():
    record = ServiceChunkRecord(
        id=1,
        collection_id="col-1",
        document_id="doc-1",
        chunk_index=0,
        content="hello",
        token_count=None,
        metadata={"source": "unit"},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    data = record.to_api_dict()
    assert data["token_count"] >= 1
    assert data["content"] == "hello"


def test_service_chunk_list_to_api_model():
    records = [
        ServiceChunkRecord(
            id=1,
            collection_id="col-1",
            document_id="doc-1",
            chunk_index=0,
            content="hello",
            token_count=10,
            metadata={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    ]

    dto = ServiceChunkList(chunks=records, total=1, page=1, page_size=10)
    api_model = dto.to_api_model()
    assert api_model.total == 1
    assert api_model.has_next is False


def test_service_global_metrics_to_api_model():
    metrics = ServiceGlobalMetrics(
        total_collections_processed=2,
        total_chunks_created=100,
        total_documents_processed=10,
        avg_chunks_per_document=10.0,
        most_used_strategy=ChunkingStrategy.RECURSIVE,
        avg_processing_time=12.5,
        success_rate=0.9,
        period_start=datetime.now(UTC),
        period_end=datetime.now(UTC),
    )
    api_model = metrics.to_api_model()
    assert api_model.most_used_strategy == ChunkingStrategy.RECURSIVE
    assert api_model.success_rate == 0.9


def test_service_quality_analysis_to_api_model():
    analysis = ServiceQualityAnalysis(
        overall_quality="good",
        quality_score=0.8,
        coherence_score=0.75,
        completeness_score=0.7,
        size_consistency=0.72,
        recommendations=["tweak"],
        issues_detected=["variance"],
    )
    api_model = analysis.to_api_model()
    assert api_model.overall_quality.value == "good"
    assert api_model.recommendations == ["tweak"]


def test_service_document_analysis_to_api_model():
    recommendation = ServiceStrategyRecommendation(
        strategy=ChunkingStrategy.RECURSIVE,
        confidence=0.9,
        reasoning="best",
        alternatives=[ChunkingStrategy.MARKDOWN],
        chunk_size=800,
        chunk_overlap=80,
    )
    analysis = ServiceDocumentAnalysis(
        document_type="txt",
        content_structure={"paragraphs": 3},
        recommended_strategy=recommendation,
        estimated_chunks={ChunkingStrategy.RECURSIVE: 3},
        complexity_score=0.6,
        special_considerations=["tables"],
    )
    api_model = analysis.to_api_model()
    assert api_model.recommended_strategy.recommended_strategy == ChunkingStrategy.RECURSIVE
    assert api_model.estimated_chunks[ChunkingStrategy.RECURSIVE] == 3


def test_service_saved_configuration_to_api_model():
    saved = ServiceSavedConfiguration(
        id="cfg-1",
        name="config",
        description="desc",
        strategy=ChunkingStrategy.RECURSIVE,
        config={"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
        created_by=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        usage_count=2,
        is_default=True,
        tags=["docs"],
    )
    api_model = saved.to_api_model()
    assert api_model.is_default is True
    assert api_model.config.chunk_size == 1000
