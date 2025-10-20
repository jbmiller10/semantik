#!/usr/bin/env python3

"""
Test UTF-8 boundary handling in streaming processor.

This module tests the critical UTF-8 boundary detection to ensure
we never split multi-byte characters.
"""

import random
import time

import pytest

from packages.shared.chunking.infrastructure.streaming.processor import StreamingDocumentProcessor
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


class TestUTF8BoundaryHandling:
    """Test suite for UTF-8 boundary detection."""

    def test_ascii_only_boundary(self) -> None:
        """Test boundary detection with ASCII-only text."""
        processor = StreamingDocumentProcessor()

        # Pure ASCII text
        data = b"Hello, World! This is ASCII text."

        # Should return full length since all ASCII
        boundary = processor._find_utf8_boundary(data)
        assert boundary == len(data)

    def test_2byte_utf8_boundary(self) -> None:
        """Test boundary detection with 2-byte UTF-8 characters."""
        processor = StreamingDocumentProcessor()

        # Text with 2-byte characters (e.g., é = 0xC3 0xA9)
        data = "Café".encode()  # C a f é -> C a f [C3 A9]

        # Test cutting in middle of é
        boundary = processor._find_utf8_boundary(data, max_pos=4)  # Would split é
        assert boundary == 3  # Should cut before é

        # Test complete character
        boundary = processor._find_utf8_boundary(data, max_pos=5)
        assert boundary == 5  # Can include complete é

    def test_3byte_utf8_boundary(self) -> None:
        """Test boundary detection with 3-byte UTF-8 characters."""
        processor = StreamingDocumentProcessor()

        # Chinese character 中 (0xE4 0xB8 0xAD)
        data = "Hello中文".encode()

        # Find where "中" starts
        hello_len = len(b"Hello")  # 5 bytes

        # Test cutting in middle of 中
        boundary = processor._find_utf8_boundary(data, max_pos=hello_len + 1)
        assert boundary == hello_len  # Should cut before 中

        # Test complete character
        boundary = processor._find_utf8_boundary(data, max_pos=hello_len + 3)
        assert boundary == hello_len + 3  # Include complete 中

    def test_4byte_utf8_boundary(self) -> None:
        """Test boundary detection with 4-byte UTF-8 characters (emoji)."""
        processor = StreamingDocumentProcessor()

        # Emoji 🎉 (F0 9F 8E 89)
        data = "Party🎉Time".encode()

        party_len = len(b"Party")  # 5 bytes

        # Test cutting in middle of emoji
        boundary = processor._find_utf8_boundary(data, max_pos=party_len + 2)
        assert boundary == party_len  # Should cut before emoji

        # Test complete emoji
        boundary = processor._find_utf8_boundary(data, max_pos=party_len + 4)
        assert boundary == party_len + 4  # Include complete emoji

    def test_mixed_utf8_boundary(self) -> None:
        """Test boundary detection with mixed UTF-8 character sizes."""
        processor = StreamingDocumentProcessor()

        # Mix of ASCII, 2-byte, 3-byte, and 4-byte characters
        text = "Hello café 中文 🎉!"
        data = text.encode("utf-8")

        # Test various positions
        for i in range(len(data)):
            boundary = processor._find_utf8_boundary(data, max_pos=i)
            # Boundary should never split a character
            if boundary > 0:
                # Should be able to decode up to boundary
                decoded = data[:boundary].decode("utf-8", errors="strict")
                assert isinstance(decoded, str)  # Should not raise

    def test_incomplete_sequence_at_end(self) -> None:
        """Test handling of incomplete UTF-8 sequence at buffer end."""
        processor = StreamingDocumentProcessor()

        # Complete text with 3-byte character at end
        complete = "Test中".encode()

        # Simulate incomplete read (missing last byte of 中)
        incomplete = complete[:-1]

        boundary = processor._find_utf8_boundary(incomplete)
        assert boundary == len(b"Test")  # Should cut before incomplete char

    def test_continuation_bytes_only(self) -> None:
        """Test handling when buffer starts with continuation bytes."""
        processor = StreamingDocumentProcessor()

        # Continuation bytes (10xxxxxx pattern)
        continuation = bytes([0x80, 0x81, 0x82])

        boundary = processor._find_utf8_boundary(continuation)
        assert boundary == 0  # No valid boundary found

    def test_streaming_window_utf8_safety(self) -> None:
        """Test StreamingWindow's UTF-8 safety in decode_safe."""
        window = StreamingWindow(max_size=1024)

        # Add text that would split a character
        text = "Hello 世界"  # 世 and 界 are 3-byte chars
        data = text.encode("utf-8")

        # Split in middle of first Chinese character
        hello_len = len(b"Hello ")
        first_part = data[: hello_len + 1]  # Incomplete 世
        second_part = data[hello_len + 1 :]

        window.append(first_part)
        decoded = window.decode_safe()

        # Should only decode "Hello " and save incomplete bytes
        assert decoded == "Hello "
        assert len(window._pending_bytes) > 0

        # Add rest and decode
        window.append(second_part)
        decoded = window.decode_safe()
        assert "世界" in decoded

    def test_random_boundaries_never_corrupt(self) -> None:
        """Test that random boundaries never cause UTF-8 corruption."""
        processor = StreamingDocumentProcessor()

        # Generate text with various UTF-8 characters
        test_strings = [
            "Simple ASCII text",
            "Café résumé naïve",  # 2-byte chars
            "中文测试文本",  # 3-byte chars
            "Emoji test 🎉🎊🎈",  # 4-byte chars
            "Mixed: café 中文 🎉",  # Mixed sizes
        ]

        for text in test_strings:
            data = text.encode("utf-8")

            # Test 100 random boundaries
            for _ in range(100):
                pos = random.randint(0, len(data))
                boundary = processor._find_utf8_boundary(data, max_pos=pos)

                # Verify we can decode up to boundary
                if boundary > 0:
                    try:
                        decoded = data[:boundary].decode("utf-8", errors="strict")
                        assert isinstance(decoded, str)
                    except UnicodeDecodeError:
                        pytest.fail(f"UTF-8 corruption at boundary {boundary} for text: {text}")

    def test_zero_length_input(self) -> None:
        """Test boundary detection with empty input."""
        processor = StreamingDocumentProcessor()

        assert processor._find_utf8_boundary(b"") == 0
        assert processor._find_utf8_boundary(b"", max_pos=0) == 0

    def test_single_byte_input(self) -> None:
        """Test boundary detection with single byte."""
        processor = StreamingDocumentProcessor()

        # ASCII byte
        assert processor._find_utf8_boundary(b"A") == 1

        # Start of 2-byte sequence (incomplete)
        assert processor._find_utf8_boundary(bytes([0xC3])) == 0

        # Continuation byte (invalid start)
        assert processor._find_utf8_boundary(bytes([0x80])) == 0

    def test_boundary_at_exact_character(self) -> None:
        """Test that boundaries align with character boundaries."""
        processor = StreamingDocumentProcessor()

        # Text with known character boundaries
        text = "A·B·C"  # · is 2-byte (0xC2 0xB7)
        data = text.encode("utf-8")

        # Character boundaries are at: 0, 1, 3, 4, 6, 7
        expected_boundaries = [0, 1, 3, 4, 6, 7]

        for i in range(len(data) + 1):
            boundary = processor._find_utf8_boundary(data, max_pos=i)
            # Boundary should be at or before requested position
            assert boundary <= i
            # And should be at a character boundary
            if boundary > 0:
                assert boundary in expected_boundaries or boundary == len(data)

    def test_malformed_utf8_handling(self) -> None:
        """Test handling of malformed UTF-8 sequences."""
        processor = StreamingDocumentProcessor()

        # Invalid UTF-8 sequences
        invalid_sequences = [
            bytes([0xFF, 0xFE]),  # Invalid start bytes
            bytes([0xC3, 0x28]),  # Invalid continuation
            bytes([0xE2, 0x82, 0x28]),  # Incomplete 3-byte
        ]

        for seq in invalid_sequences:
            boundary = processor._find_utf8_boundary(seq)
            # Should handle gracefully without crashing
            assert boundary >= 0
            assert boundary <= len(seq)


class TestUTF8BoundaryPerformance:
    """Performance tests for UTF-8 boundary detection."""

    def test_large_ascii_performance(self) -> None:
        """Test performance with large ASCII-only data."""
        processor = StreamingDocumentProcessor()

        # 1MB of ASCII data
        data = b"A" * (1024 * 1024)

        start = time.time()
        boundary = processor._find_utf8_boundary(data)
        elapsed = time.time() - start

        assert boundary == len(data)
        assert elapsed < 0.01  # Should be very fast for ASCII

    def test_large_mixed_utf8_performance(self) -> None:
        """Test performance with large mixed UTF-8 data."""
        processor = StreamingDocumentProcessor()

        # Generate mixed UTF-8 data
        text = "Hello 世界 " * 10000  # ~100KB
        data = text.encode("utf-8")

        start = time.time()

        # Test many boundary detections
        for i in range(0, len(data), 1000):
            boundary = processor._find_utf8_boundary(data, max_pos=i)
            assert boundary <= i

        elapsed = time.time() - start
        assert elapsed < 1.0  # Should complete within 1 second
