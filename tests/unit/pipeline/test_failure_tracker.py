"""Unit tests for consecutive failure tracker."""

import pytest

from shared.pipeline.failure_tracker import ConsecutiveFailureTracker, FailureRecord


class TestFailureRecord:
    """Tests for FailureRecord dataclass."""

    def test_creation(self) -> None:
        """Test creating a FailureRecord."""
        record = FailureRecord(
            file_uri="file:///test.pdf",
            stage_id="parser-1",
            error_type="ParserError",
            error_message="Parse failed",
        )
        assert record.file_uri == "file:///test.pdf"
        assert record.stage_id == "parser-1"
        assert record.error_type == "ParserError"
        assert record.error_message == "Parse failed"
        assert record.timestamp is not None


class TestConsecutiveFailureTracker:
    """Tests for ConsecutiveFailureTracker class."""

    def test_init_default_threshold(self) -> None:
        """Test default threshold is 10."""
        tracker = ConsecutiveFailureTracker()
        assert tracker.threshold == 10

    def test_init_custom_threshold(self) -> None:
        """Test custom threshold."""
        tracker = ConsecutiveFailureTracker(threshold=5)
        assert tracker.threshold == 5

    def test_init_invalid_threshold(self) -> None:
        """Test that threshold must be at least 1."""
        with pytest.raises(ValueError, match="threshold must be at least 1"):
            ConsecutiveFailureTracker(threshold=0)

    def test_initial_state(self) -> None:
        """Test initial state is clean."""
        tracker = ConsecutiveFailureTracker()
        assert tracker.consecutive_count == 0
        assert tracker.recent_failures == []
        assert not tracker.should_halt()

    def test_record_failure_increments_count(self) -> None:
        """Test recording failure increments consecutive count."""
        tracker = ConsecutiveFailureTracker()

        tracker.record_failure("file:///test.pdf", "parser", "Error")

        assert tracker.consecutive_count == 1
        assert len(tracker.recent_failures) == 1

    def test_record_success_resets_count(self) -> None:
        """Test recording success resets consecutive count."""
        tracker = ConsecutiveFailureTracker()
        tracker.record_failure("file:///a.pdf", "parser", "Error")
        tracker.record_failure("file:///b.pdf", "parser", "Error")
        assert tracker.consecutive_count == 2

        tracker.record_success()

        assert tracker.consecutive_count == 0

    def test_success_does_not_clear_failure_records(self) -> None:
        """Test success resets count but keeps failure records."""
        tracker = ConsecutiveFailureTracker()
        tracker.record_failure("file:///a.pdf", "parser", "Error")

        tracker.record_success()

        assert tracker.consecutive_count == 0
        assert len(tracker.recent_failures) == 1  # Record still there

    def test_should_halt_at_threshold(self) -> None:
        """Test should_halt returns True when threshold reached."""
        tracker = ConsecutiveFailureTracker(threshold=3)

        # Record failures below threshold
        tracker.record_failure("file:///a.pdf", "parser", "Error")
        tracker.record_failure("file:///b.pdf", "parser", "Error")
        assert not tracker.should_halt()

        # Record failure to reach threshold
        tracker.record_failure("file:///c.pdf", "parser", "Error")
        assert tracker.should_halt()

    def test_should_halt_above_threshold(self) -> None:
        """Test should_halt stays True above threshold."""
        tracker = ConsecutiveFailureTracker(threshold=2)

        tracker.record_failure("file:///a.pdf", "parser", "Error")
        tracker.record_failure("file:///b.pdf", "parser", "Error")
        tracker.record_failure("file:///c.pdf", "parser", "Error")

        assert tracker.should_halt()
        assert tracker.consecutive_count == 3

    def test_get_halt_reason_when_halted(self) -> None:
        """Test get_halt_reason returns message when halted."""
        tracker = ConsecutiveFailureTracker(threshold=2)
        tracker.record_failure("file:///a.pdf", "parser", "Error", error_type="ParserError")
        tracker.record_failure("file:///b.pdf", "chunker", "Error", error_type="ChunkError")

        reason = tracker.get_halt_reason()

        assert "2 consecutive failures" in reason
        assert "parser" in reason or "chunker" in reason

    def test_get_halt_reason_when_not_halted(self) -> None:
        """Test get_halt_reason returns empty string when not halted."""
        tracker = ConsecutiveFailureTracker(threshold=10)
        tracker.record_failure("file:///a.pdf", "parser", "Error")

        reason = tracker.get_halt_reason()

        assert reason == ""

    def test_reset_clears_everything(self) -> None:
        """Test reset clears all state."""
        tracker = ConsecutiveFailureTracker(threshold=5)
        tracker.record_failure("file:///a.pdf", "parser", "Error")
        tracker.record_failure("file:///b.pdf", "parser", "Error")

        tracker.reset()

        assert tracker.consecutive_count == 0
        assert tracker.recent_failures == []
        assert not tracker.should_halt()

    def test_failure_records_have_timestamps(self) -> None:
        """Test failure records include timestamps."""
        tracker = ConsecutiveFailureTracker()
        tracker.record_failure("file:///test.pdf", "parser", "Error")

        record = tracker.recent_failures[0]

        assert record.timestamp is not None

    def test_recent_failures_trimmed(self) -> None:
        """Test recent failures list is trimmed to limit."""
        tracker = ConsecutiveFailureTracker(threshold=100, recent_limit=3)

        # Record more than the limit
        for i in range(5):
            tracker.record_failure(f"file:///{i}.pdf", "parser", f"Error {i}")

        # Should only keep the most recent 3
        assert len(tracker.recent_failures) == 3
        # Most recent should be the last one added
        assert tracker.recent_failures[-1].file_uri == "file:///4.pdf"

    def test_error_type_recorded(self) -> None:
        """Test error type is recorded in failure record."""
        tracker = ConsecutiveFailureTracker()
        tracker.record_failure(
            "file:///test.pdf",
            "parser",
            "Parse failed",
            error_type="UnsupportedFormatError",
        )

        record = tracker.recent_failures[0]

        assert record.error_type == "UnsupportedFormatError"

    def test_default_error_type(self) -> None:
        """Test default error type is 'unknown'."""
        tracker = ConsecutiveFailureTracker()
        tracker.record_failure("file:///test.pdf", "parser", "Error")

        record = tracker.recent_failures[0]

        assert record.error_type == "unknown"

    def test_mixed_success_failure_pattern(self) -> None:
        """Test mixed pattern of successes and failures."""
        tracker = ConsecutiveFailureTracker(threshold=3)

        # Fail twice, succeed, fail once
        tracker.record_failure("file:///a.pdf", "parser", "Error")
        tracker.record_failure("file:///b.pdf", "parser", "Error")
        assert tracker.consecutive_count == 2

        tracker.record_success()
        assert tracker.consecutive_count == 0

        tracker.record_failure("file:///c.pdf", "parser", "Error")
        assert tracker.consecutive_count == 1

        assert not tracker.should_halt()
