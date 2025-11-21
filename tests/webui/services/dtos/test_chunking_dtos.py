"""
Unit tests for chunking service DTOs.

Tests all DTO classes and their to_api_model() conversion methods,
including error handling and edge cases.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from webui.services.dtos.api_models import (
    ChunkingConfigBase,
    ChunkingStats,
    ChunkingStrategy,
    ChunkPreview,
    CompareResponse,
    PreviewResponse,
    StrategyInfo,
    StrategyRecommendation,
)
from webui.services.dtos.chunking_dtos import (
    ServiceChunkingStats,
    ServiceChunkPreview,
    ServiceCompareResponse,
    ServicePreviewResponse,
    ServiceStrategyComparison,
    ServiceStrategyInfo,
    ServiceStrategyMetrics,
    ServiceStrategyRecommendation,
)


class TestServiceChunkPreview:
    """Test ServiceChunkPreview DTO."""

    def test_to_api_model_with_all_fields(self):
        """Test conversion when all fields are provided."""
        dto = ServiceChunkPreview(
            index=0,
            content="Test content here",
            token_count=10,
            char_count=17,
            quality_score=0.9,
            metadata={"key": "value"},
            overlap_info={"prev": 5, "next": 3},
        )

        result = dto.to_api_model()

        assert isinstance(result, ChunkPreview)
        assert result.index == 0
        assert result.content == "Test content here"
        assert result.token_count == 10
        assert result.char_count == 17  # Always recalculated as len(content)
        assert result.quality_score == 0.9
        assert result.metadata == {"key": "value"}
        assert result.overlap_info == {"prev": 5, "next": 3}

    def test_to_api_model_calculates_char_count(self):
        """Test that char_count is always calculated from content length."""
        dto = ServiceChunkPreview(index=0, content="Test", char_count=999)  # char_count should be ignored

        result = dto.to_api_model()

        assert result.char_count == 4  # len("Test")

    def test_to_api_model_estimates_token_count(self):
        """Test token_count estimation when not provided."""
        dto = ServiceChunkPreview(index=0, content="Test content here now")  # 21 chars

        result = dto.to_api_model()

        # Should estimate as char_count // 4
        assert result.token_count == 21 // 4  # = 5

    def test_to_api_model_with_text_field(self):
        """Test handling of 'text' field instead of 'content'."""
        dto = ServiceChunkPreview(index=1, text="Text field content")

        result = dto.to_api_model()

        assert result.content == "Text field content"
        assert result.char_count == 18

    def test_to_api_model_default_quality_score(self):
        """Test default quality_score is 0.8."""
        dto = ServiceChunkPreview(index=0, content="Test")

        result = dto.to_api_model()

        assert result.quality_score == 0.8

    def test_to_api_model_empty_content(self):
        """Test handling of empty content."""
        dto = ServiceChunkPreview(index=0, content="")

        result = dto.to_api_model()

        assert result.content == ""
        assert result.char_count == 0
        assert result.token_count == 0


class TestServiceStrategyInfo:
    """Test ServiceStrategyInfo DTO."""

    def test_to_api_model_success(self):
        """Test successful conversion to API model."""
        dto = ServiceStrategyInfo(
            id="fixed_size",
            name="Fixed Size Chunking",
            description="Splits text into fixed-size chunks",
            best_for=["txt", "log"],
            pros=["Fast", "Predictable"],
            cons=["May split sentences"],
            default_config={"strategy": "fixed_size", "chunk_size": 512},
            performance_characteristics={"speed": "fast", "accuracy": "medium"},
        )

        result = dto.to_api_model()

        assert isinstance(result, StrategyInfo)
        assert result.id == "fixed_size"
        assert result.name == "Fixed Size Chunking"
        assert result.description == "Splits text into fixed-size chunks"
        assert result.best_for == ["txt", "log"]
        assert result.pros == ["Fast", "Predictable"]
        assert result.cons == ["May split sentences"]
        assert isinstance(result.default_config, ChunkingConfigBase)
        assert result.default_config.strategy == ChunkingStrategy.FIXED_SIZE

    def test_to_api_model_adds_missing_strategy(self):
        """Test that missing strategy in config is added from id."""
        dto = ServiceStrategyInfo(
            id="semantic",
            name="Semantic Chunking",
            description="Semantic-based chunking",
            default_config={"chunk_size": 1000},  # Missing strategy
        )

        result = dto.to_api_model()

        assert result.default_config.strategy == ChunkingStrategy.SEMANTIC

    @patch("webui.services.dtos.chunking_dtos.logger")
    def test_to_api_model_validation_error_handling(self, mock_logger):
        """Test handling of ValidationError in config creation."""
        dto = ServiceStrategyInfo(
            id="fixed_size",
            name="Fixed Size",
            description="Test",
            default_config={"chunk_size": 10},  # Too small - ge=100
        )

        result = dto.to_api_model()

        # Should log warning and use fallback config
        mock_logger.warning.assert_called()
        assert result.default_config.strategy == ChunkingStrategy.FIXED_SIZE


class TestServicePreviewResponse:
    """Test ServicePreviewResponse DTO."""

    def test_to_api_model_with_dto_chunks(self):
        """Test conversion with ServiceChunkPreview DTOs."""
        chunk1 = ServiceChunkPreview(index=0, content="Chunk 1")
        chunk2 = ServiceChunkPreview(index=1, content="Chunk 2")

        dto = ServicePreviewResponse(
            preview_id="test-123",
            strategy="fixed_size",
            config={"strategy": "fixed_size", "chunk_size": 512},
            chunks=[chunk1, chunk2],
            total_chunks=2,
            processing_time_ms=100,
        )

        result = dto.to_api_model()

        assert isinstance(result, PreviewResponse)
        assert result.preview_id == "test-123"
        assert result.strategy == ChunkingStrategy.FIXED_SIZE
        assert len(result.chunks) == 2
        assert all(isinstance(chunk, ChunkPreview) for chunk in result.chunks)

    def test_to_api_model_with_dict_chunks(self):
        """Test conversion with dict chunks from service layer."""
        dto = ServicePreviewResponse(
            preview_id="test-456",
            strategy="semantic",
            config={"chunk_size": 1000},
            chunks=[
                {"index": 0, "content": "Dict chunk 1"},
                {"index": 1, "text": "Dict chunk 2"},  # Using 'text' field
            ],
            total_chunks=2,
            processing_time_ms=150,
        )

        result = dto.to_api_model()

        assert result.strategy == ChunkingStrategy.SEMANTIC
        assert len(result.chunks) == 2
        assert result.chunks[0].content == "Dict chunk 1"
        assert result.chunks[1].content == "Dict chunk 2"

    def test_to_api_model_default_expires_at(self):
        """Test default expires_at is set to 15 minutes from now."""
        dto = ServicePreviewResponse(
            preview_id="test-789",
            strategy="recursive",
            config={},
            chunks=[],
            total_chunks=0,
            processing_time_ms=50,
        )

        before = datetime.now(UTC)
        result = dto.to_api_model()
        after = datetime.now(UTC)

        # Check expires_at is approximately 15 minutes from now
        expected_min = before + timedelta(minutes=15)
        expected_max = after + timedelta(minutes=15)
        assert expected_min <= result.expires_at <= expected_max

    def test_to_api_model_cached_flag(self):
        """Test cached flag is passed through."""
        dto = ServicePreviewResponse(
            preview_id="cached-123",
            strategy="fixed_size",
            config={},
            chunks=[],
            total_chunks=0,
            processing_time_ms=1,
            cached=True,
        )

        result = dto.to_api_model()

        assert result.cached is True

    @patch("webui.services.dtos.chunking_dtos.logger")
    def test_to_api_model_config_validation_error(self, mock_logger):
        """Test handling of config validation error."""
        dto = ServicePreviewResponse(
            preview_id="error-123",
            strategy="fixed_size",
            config={"chunk_overlap": 5000},  # Too large - le=500
            chunks=[],
            total_chunks=0,
            processing_time_ms=100,
        )

        result = dto.to_api_model()

        # Should log warning and use fallback config
        mock_logger.warning.assert_called()
        assert result.config.strategy == ChunkingStrategy.FIXED_SIZE


class TestServiceStrategyRecommendation:
    """Test ServiceStrategyRecommendation DTO."""

    def test_to_api_model_string_strategy(self):
        """Test conversion with string strategy."""
        dto = ServiceStrategyRecommendation(
            strategy="semantic",
            confidence=0.85,
            reasoning="Document has clear semantic boundaries",
            alternatives=["recursive", "markdown"],
            chunk_size=1024,
            chunk_overlap=100,
        )

        result = dto.to_api_model()

        assert isinstance(result, StrategyRecommendation)
        assert result.recommended_strategy == ChunkingStrategy.SEMANTIC
        assert result.confidence == 0.85
        assert result.reasoning == "Document has clear semantic boundaries"
        assert result.alternative_strategies == [
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.MARKDOWN,
        ]
        assert result.suggested_config.chunk_size == 1024
        assert result.suggested_config.chunk_overlap == 100

    def test_to_api_model_enum_strategy(self):
        """Test conversion with enum strategy."""
        dto = ServiceStrategyRecommendation(
            strategy=ChunkingStrategy.HIERARCHICAL,
            confidence=0.95,
            reasoning="Complex document structure",
        )

        result = dto.to_api_model()

        assert result.recommended_strategy == ChunkingStrategy.HIERARCHICAL
        assert result.suggested_config.strategy == ChunkingStrategy.HIERARCHICAL

    def test_to_api_model_default_values(self):
        """Test default values for optional fields."""
        dto = ServiceStrategyRecommendation(strategy="fixed_size", confidence=0.7, reasoning="Simple text")

        result = dto.to_api_model()

        assert result.alternative_strategies == []
        assert result.suggested_config.chunk_size == 512  # Default
        assert result.suggested_config.chunk_overlap == 50  # Default


class TestServiceStrategyComparison:
    """Test ServiceStrategyComparison DTO."""

    def test_to_api_model_with_sample_chunks(self):
        """Test conversion with sample chunks."""
        chunk1 = ServiceChunkPreview(index=0, content="Sample 1")
        chunk2 = ServiceChunkPreview(index=1, content="Sample 2")

        dto = ServiceStrategyComparison(
            strategy="fixed_size",
            config={"chunk_size": 512},
            sample_chunks=[chunk1, chunk2],
            total_chunks=10,
            avg_chunk_size=500,
            size_variance=50.0,
            quality_score=0.85,
            processing_time_ms=250,
            pros=["Fast"],
            cons=["May split sentences"],
        )

        result = dto.to_api_model()

        assert result.strategy == ChunkingStrategy.FIXED_SIZE
        assert result.total_chunks == 10
        assert result.avg_chunk_size == 500
        assert len(result.sample_chunks) == 2
        assert all(isinstance(chunk, ChunkPreview) for chunk in result.sample_chunks)

    def test_to_api_model_with_dict_chunks(self):
        """Test conversion with dict chunks."""
        dto = ServiceStrategyComparison(
            strategy="semantic",
            config={},
            sample_chunks=[{"index": 0, "content": "Dict sample"}],
            total_chunks=8,
            avg_chunk_size=600,
            size_variance=25.0,
            quality_score=0.9,
            processing_time_ms=300,
        )

        result = dto.to_api_model()

        assert result.strategy == ChunkingStrategy.SEMANTIC
        assert len(result.sample_chunks) == 1
        assert result.sample_chunks[0].content == "Dict sample"


class TestServiceCompareResponse:
    """Test ServiceCompareResponse DTO."""

    def test_to_api_model_complete(self):
        """Test complete conversion with all fields."""
        comparison1 = ServiceStrategyComparison(
            strategy="fixed_size",
            config={},
            sample_chunks=[],
            total_chunks=10,
            avg_chunk_size=500,
            size_variance=20.0,
            quality_score=0.8,
            processing_time_ms=100,
        )

        comparison2 = ServiceStrategyComparison(
            strategy="semantic",
            config={},
            sample_chunks=[],
            total_chunks=8,
            avg_chunk_size=600,
            size_variance=30.0,
            quality_score=0.9,
            processing_time_ms=150,
        )

        dto = ServiceCompareResponse(
            comparison_id="compare-123",
            comparisons=[comparison1, comparison2],
            recommendation=ServiceStrategyRecommendation(
                strategy="semantic", confidence=0.9, reasoning="Best for this document type"
            ),
            processing_time_ms=250,
        )

        result = dto.to_api_model()

        assert isinstance(result, CompareResponse)
        assert result.comparison_id == "compare-123"
        assert len(result.comparisons) == 2
        assert result.recommendation.recommended_strategy == ChunkingStrategy.SEMANTIC
        assert result.processing_time_ms == 250

    def test_to_api_model_with_dict_comparisons(self):
        """Test conversion with dict comparisons from service layer."""
        dto = ServiceCompareResponse(
            comparison_id="dict-test",
            comparisons=[
                {
                    "strategy": "fixed_size",
                    "config": {"chunk_size": 512},
                    "sample_chunks": [],
                    "total_chunks": 10,
                    "avg_chunk_size": 500.0,
                    "size_variance": 20.0,
                    "quality_score": 0.8,
                    "processing_time_ms": 100,
                }
            ],
            recommendation=ServiceStrategyRecommendation(strategy="fixed_size", confidence=0.9, reasoning="Test"),
            processing_time_ms=100,
        )

        result = dto.to_api_model()
        assert len(result.comparisons) == 1
        assert result.comparisons[0].strategy == ChunkingStrategy.FIXED_SIZE

    def test_to_api_model_with_dict_recommendation(self):
        """Test conversion with dict recommendation from service layer."""
        dto = ServiceCompareResponse(
            comparison_id="dict-rec-test",
            comparisons=[],
            recommendation={
                "strategy": "semantic",
                "confidence": 0.85,
                "reasoning": "Best for document",
                "alternatives": ["recursive"],
                "chunk_size": 1024,
                "chunk_overlap": 100,
            },
            processing_time_ms=150,
        )

        result = dto.to_api_model()
        assert result.recommendation.recommended_strategy == ChunkingStrategy.SEMANTIC
        assert result.recommendation.confidence == 0.85

    def test_to_api_model_with_invalid_recommendation(self):
        """Test fallback when recommendation is neither DTO nor dict."""
        dto = ServiceCompareResponse(
            comparison_id="fallback-test",
            comparisons=[],
            recommendation="invalid_type",  # Invalid type
            processing_time_ms=100,
        )

        result = dto.to_api_model()
        # Should use fallback recommendation
        assert result.recommendation.recommended_strategy == ChunkingStrategy.FIXED_SIZE
        assert result.recommendation.confidence == 0.5
        assert "Unable to determine" in result.recommendation.reasoning


class TestServiceStrategyMetrics:
    """Test ServiceStrategyMetrics DTO."""

    def test_to_api_model(self):
        """Test conversion to API model."""
        dto = ServiceStrategyMetrics(
            strategy="fixed_size",
            usage_count=100,
            avg_chunk_size=512,
            avg_processing_time=2.5,
            success_rate=0.95,
            avg_quality_score=0.85,
            best_for_types=["txt", "log"],
        )

        result = dto.to_api_model()

        assert result.strategy == ChunkingStrategy.FIXED_SIZE
        assert result.usage_count == 100
        assert result.avg_chunk_size == 512
        assert result.avg_processing_time == 2.5
        assert result.success_rate == 0.95
        assert result.avg_quality_score == 0.85
        assert result.best_for_types == ["txt", "log"]

    def test_create_default_metrics(self):
        """Test creation of default metrics."""
        defaults = ServiceStrategyMetrics.create_default_metrics()

        assert len(defaults) == 6  # Six primary strategies
        strategies = [m.strategy for m in defaults]
        assert "fixed_size" in strategies
        assert "semantic" in strategies
        assert "recursive" in strategies
        assert "markdown" in strategies
        assert "hierarchical" in strategies
        assert "hybrid" in strategies

        # All should have zero usage
        for metric in defaults:
            assert metric.usage_count == 0
            assert metric.success_rate == 0.0


class TestServiceChunkingStats:
    """Test ServiceChunkingStats DTO."""

    def test_to_api_model(self):
        """Test conversion to API model."""
        dto = ServiceChunkingStats(
            total_chunks=100,
            total_documents=10,
            avg_chunk_size=500,
            min_chunk_size=100,
            max_chunk_size=1000,
            size_variance=50.5,
            strategy_used="recursive",
            last_updated=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            processing_time_seconds=5.5,
            quality_metrics={"speed": "fast", "quality": "high"},
        )

        result = dto.to_api_model()

        assert isinstance(result, ChunkingStats)
        assert result.total_chunks == 100
        assert result.total_documents == 10
        assert result.avg_chunk_size == 500
        assert result.min_chunk_size == 100
        assert result.max_chunk_size == 1000
        assert result.size_variance == 50.5
        assert result.strategy_used == ChunkingStrategy.RECURSIVE
        assert result.last_updated == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        assert result.processing_time_seconds == 5.5
        assert result.quality_metrics == {"speed": "fast", "quality": "high"}

    def test_to_api_model_with_defaults(self):
        """Test conversion with default values."""
        dto = ServiceChunkingStats(strategy_used="fixed_size")

        result = dto.to_api_model()

        assert result.total_chunks == 0
        assert result.total_documents == 0
        assert result.avg_chunk_size == 0
        assert result.strategy_used == ChunkingStrategy.FIXED_SIZE
        assert isinstance(result.last_updated, datetime)
        assert result.processing_time_seconds == 0.0
        assert result.quality_metrics == {}


class TestErrorHandling:
    """Test error handling across all DTOs."""

    @patch("webui.services.dtos.chunking_dtos.logger")
    def test_logging_on_validation_errors(self, mock_logger):
        """Test that validation errors are logged appropriately."""
        # Create DTO with invalid config that will cause ValidationError
        dto = ServicePreviewResponse(
            preview_id="test",
            strategy="fixed_size",  # Valid strategy
            config={"chunk_size": "not_an_int"},  # Invalid config - chunk_size should be int
            chunks=[],
            total_chunks=0,
            processing_time_ms=100,
        )

        # Should not raise, but log warning for invalid config
        result = dto.to_api_model()

        # Check that a warning was logged about invalid config
        mock_logger.warning.assert_called()
        assert result.config.strategy == ChunkingStrategy.FIXED_SIZE

    @patch("webui.services.dtos.chunking_dtos.logger")
    def test_invalid_strategy_conversions(self, mock_logger):
        """Test handling of invalid strategy values across DTOs."""
        # Test ServicePreviewResponse with invalid strategy
        dto1 = ServicePreviewResponse(
            preview_id="invalid-strategy",
            strategy="invalid_strategy_name",
            config={},
            chunks=[],
            total_chunks=0,
            processing_time_ms=100,
        )
        result1 = dto1.to_api_model()
        mock_logger.warning.assert_called()
        assert result1.strategy == ChunkingStrategy.FIXED_SIZE  # Fallback

        # Reset mock
        mock_logger.reset_mock()

        # Test ServiceStrategyMetrics with invalid strategy
        dto2 = ServiceStrategyMetrics(strategy="non_existent_strategy", usage_count=10, avg_chunk_size=500)
        result2 = dto2.to_api_model()
        mock_logger.warning.assert_called()
        assert result2.strategy == ChunkingStrategy.FIXED_SIZE  # Fallback

    @patch("webui.services.dtos.chunking_dtos.logger")
    def test_invalid_alternative_strategies(self, mock_logger):
        """Test handling of invalid alternative strategies."""
        dto = ServiceStrategyRecommendation(
            strategy="semantic",
            confidence=0.8,
            reasoning="Test",
            alternatives=["recursive", "invalid_strategy", "markdown", "another_invalid"],
        )
        result = dto.to_api_model()

        # Should have logged warnings for invalid strategies
        assert mock_logger.warning.call_count >= 2  # At least 2 warnings for invalid alternatives

        # Should only include valid alternatives
        assert ChunkingStrategy.RECURSIVE in result.alternative_strategies
        assert ChunkingStrategy.MARKDOWN in result.alternative_strategies
        assert len(result.alternative_strategies) == 2  # Only recursive and markdown are valid


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_chunk_content_vs_text_field(self):
        """Test that both 'content' and 'text' fields work."""
        dto1 = ServiceChunkPreview(index=0, content="Using content field")
        dto2 = ServiceChunkPreview(index=1, text="Using text field")

        result1 = dto1.to_api_model()
        result2 = dto2.to_api_model()

        assert result1.content == "Using content field"
        assert result2.content == "Using text field"

    def test_mixed_chunk_types_in_preview(self):
        """Test PreviewResponse handles mixed chunk types."""
        dto = ServicePreviewResponse(
            preview_id="mixed",
            strategy="fixed_size",
            config={},
            chunks=[
                ServiceChunkPreview(index=0, content="DTO chunk"),
                {"index": 1, "content": "Dict chunk"},
                {"index": 2, "text": "Dict with text field"},
            ],
            total_chunks=3,
            processing_time_ms=100,
        )

        result = dto.to_api_model()

        assert len(result.chunks) == 3
        assert result.chunks[0].content == "DTO chunk"
        assert result.chunks[1].content == "Dict chunk"
        assert result.chunks[2].content == "Dict with text field"
