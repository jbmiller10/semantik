"""Unit tests for pipeline executor types."""

import pytest

from shared.pipeline.executor_types import (
    ChunkStats,
    ExecutionMode,
    ExecutionResult,
    ProgressEvent,
    SampleOutput,
    StageFailure,
)
from shared.pipeline.types import FileReference


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_full_mode_value(self) -> None:
        """Test FULL mode has correct value."""
        assert ExecutionMode.FULL.value == "full"

    def test_dry_run_mode_value(self) -> None:
        """Test DRY_RUN mode has correct value."""
        assert ExecutionMode.DRY_RUN.value == "dry_run"

    def test_enum_string_conversion(self) -> None:
        """Test enum can be converted from string."""
        assert ExecutionMode("full") == ExecutionMode.FULL
        assert ExecutionMode("dry_run") == ExecutionMode.DRY_RUN

    def test_invalid_mode_raises(self) -> None:
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid"):
            ExecutionMode("invalid")


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_minimal_event(self) -> None:
        """Test creating event with minimal fields."""
        event = ProgressEvent(event_type="file_started")
        assert event.event_type == "file_started"
        assert event.file_uri is None
        assert event.stage_id is None
        assert event.details == {}

    def test_full_event(self) -> None:
        """Test creating event with all fields."""
        event = ProgressEvent(
            event_type="file_completed",
            file_uri="file:///test.pdf",
            stage_id="parser-1",
            details={"chunks_created": 5},
        )
        assert event.event_type == "file_completed"
        assert event.file_uri == "file:///test.pdf"
        assert event.stage_id == "parser-1"
        assert event.details == {"chunks_created": 5}

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        event = ProgressEvent(
            event_type="stage_completed",
            file_uri="file:///test.md",
            stage_id="chunker-1",
            details={"tokens": 500},
        )
        d = event.to_dict()
        assert d == {
            "event_type": "stage_completed",
            "file_uri": "file:///test.md",
            "stage_id": "chunker-1",
            "details": {"tokens": 500},
        }


class TestChunkStats:
    """Tests for ChunkStats dataclass."""

    def test_creation(self) -> None:
        """Test creating ChunkStats."""
        stats = ChunkStats(
            total_chunks=10,
            avg_tokens=500.5,
            min_tokens=100,
            max_tokens=1000,
        )
        assert stats.total_chunks == 10
        assert stats.avg_tokens == 500.5
        assert stats.min_tokens == 100
        assert stats.max_tokens == 1000

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        stats = ChunkStats(
            total_chunks=5,
            avg_tokens=250.0,
            min_tokens=50,
            max_tokens=500,
        )
        d = stats.to_dict()
        assert d == {
            "total_chunks": 5,
            "avg_tokens": 250.0,
            "min_tokens": 50,
            "max_tokens": 500,
        }

    def test_from_token_counts_empty(self) -> None:
        """Test from_token_counts with empty list returns None."""
        result = ChunkStats.from_token_counts([])
        assert result is None

    def test_from_token_counts_single(self) -> None:
        """Test from_token_counts with single value."""
        result = ChunkStats.from_token_counts([500])
        assert result is not None
        assert result.total_chunks == 1
        assert result.avg_tokens == 500.0
        assert result.min_tokens == 500
        assert result.max_tokens == 500

    def test_from_token_counts_multiple(self) -> None:
        """Test from_token_counts with multiple values."""
        result = ChunkStats.from_token_counts([100, 200, 300, 400, 500])
        assert result is not None
        assert result.total_chunks == 5
        assert result.avg_tokens == 300.0
        assert result.min_tokens == 100
        assert result.max_tokens == 500


class TestStageFailure:
    """Tests for StageFailure dataclass."""

    def test_creation(self) -> None:
        """Test creating StageFailure."""
        failure = StageFailure(
            file_uri="file:///bad.pdf",
            stage_id="parser-1",
            stage_type="parser",
            error_type="ParserError",
            error_message="Failed to parse PDF",
        )
        assert failure.file_uri == "file:///bad.pdf"
        assert failure.stage_id == "parser-1"
        assert failure.stage_type == "parser"
        assert failure.error_type == "ParserError"
        assert failure.error_message == "Failed to parse PDF"
        assert failure.error_traceback is None

    def test_with_traceback(self) -> None:
        """Test creating StageFailure with traceback."""
        failure = StageFailure(
            file_uri="file:///bad.pdf",
            stage_id="parser-1",
            stage_type="parser",
            error_type="Exception",
            error_message="Oops",
            error_traceback="Traceback (most recent call last):\n...",
        )
        assert failure.error_traceback == "Traceback (most recent call last):\n..."

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        failure = StageFailure(
            file_uri="file:///test.pdf",
            stage_id="chunker-1",
            stage_type="chunker",
            error_type="ValueError",
            error_message="Invalid chunk size",
        )
        d = failure.to_dict()
        assert d == {
            "file_uri": "file:///test.pdf",
            "stage_id": "chunker-1",
            "stage_type": "chunker",
            "error_type": "ValueError",
            "error_message": "Invalid chunk size",
            "error_traceback": None,
        }


class TestSampleOutput:
    """Tests for SampleOutput dataclass."""

    def test_creation(self) -> None:
        """Test creating SampleOutput."""
        file_ref = FileReference(
            uri="file:///test.md",
            source_type="directory",
            content_type="document",
            size_bytes=100,
        )
        sample = SampleOutput(
            file_ref=file_ref,
            chunks=[{"content": "Hello", "metadata": {}}],
            parse_metadata={"pages": 1},
        )
        assert sample.file_ref == file_ref
        assert len(sample.chunks) == 1
        assert sample.parse_metadata == {"pages": 1}

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=50,
        )
        sample = SampleOutput(
            file_ref=file_ref,
            chunks=[],
            parse_metadata={},
        )
        d = sample.to_dict()
        assert d["file_ref"]["uri"] == "file:///test.txt"
        assert d["chunks"] == []
        assert d["parse_metadata"] == {}


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_minimal_result(self) -> None:
        """Test creating minimal ExecutionResult."""
        result = ExecutionResult(
            mode=ExecutionMode.FULL,
            files_processed=10,
            files_succeeded=8,
            files_failed=2,
            files_skipped=0,
            chunks_created=50,
            chunk_stats=None,
            failures=[],
            stage_timings={},
            total_duration_ms=1000.0,
        )
        assert result.mode == ExecutionMode.FULL
        assert result.files_processed == 10
        assert result.files_succeeded == 8
        assert result.files_failed == 2
        assert result.files_skipped == 0
        assert result.chunks_created == 50
        assert result.halted is False
        assert result.halt_reason is None

    def test_full_result(self) -> None:
        """Test creating full ExecutionResult."""
        chunk_stats = ChunkStats(
            total_chunks=50,
            avg_tokens=200.0,
            min_tokens=50,
            max_tokens=400,
        )
        failure = StageFailure(
            file_uri="file:///bad.pdf",
            stage_id="parser-1",
            stage_type="parser",
            error_type="Error",
            error_message="Parse failed",
        )
        result = ExecutionResult(
            mode=ExecutionMode.DRY_RUN,
            files_processed=5,
            files_succeeded=4,
            files_failed=1,
            files_skipped=0,
            chunks_created=0,
            chunk_stats=chunk_stats,
            failures=[failure],
            stage_timings={"parser:parser-1": 500.0, "chunker:chunker-1": 200.0},
            total_duration_ms=800.0,
            sample_outputs=[],
            halted=True,
            halt_reason="Too many failures",
        )
        assert result.mode == ExecutionMode.DRY_RUN
        assert result.chunk_stats is not None
        assert len(result.failures) == 1
        assert result.stage_timings["parser:parser-1"] == 500.0
        assert result.halted is True
        assert result.halt_reason == "Too many failures"

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        result = ExecutionResult(
            mode=ExecutionMode.FULL,
            files_processed=2,
            files_succeeded=2,
            files_failed=0,
            files_skipped=0,
            chunks_created=10,
            chunk_stats=ChunkStats(10, 100.0, 50, 150),
            failures=[],
            stage_timings={"loader": 50.0},
            total_duration_ms=100.0,
        )
        d = result.to_dict()
        assert d["mode"] == "full"
        assert d["files_processed"] == 2
        assert d["chunk_stats"] is not None
        assert d["chunk_stats"]["total_chunks"] == 10
        assert d["failures"] == []
        assert d["halted"] is False
