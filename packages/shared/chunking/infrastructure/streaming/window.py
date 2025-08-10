#!/usr/bin/env python3
"""
Streaming window for bounded memory text processing.

This module implements a sliding window that processes documents in chunks
while maintaining bounded memory usage and respecting UTF-8 boundaries.
"""

import logging

logger = logging.getLogger(__name__)


class StreamingWindow:
    """
    A sliding window for stream processing text with bounded memory.

    The window maintains a fixed-size buffer that slides through the document,
    allowing processing of arbitrarily large files with constant memory usage.
    """

    def __init__(self, max_size: int = 256 * 1024):
        """
        Initialize the streaming window.

        Args:
            max_size: Maximum window size in bytes (default 256KB)
        """
        self.max_size = max_size
        self.buffer = bytearray()
        self.text_buffer = ""  # Decoded text buffer
        self.byte_offset = 0  # Current byte position in stream
        self.char_offset = 0  # Current character position in stream
        self._pending_bytes = bytearray()  # Incomplete UTF-8 bytes

    def append(self, data: bytes) -> None:
        """
        Append data to the window.

        Args:
            data: Raw bytes to append

        Raises:
            MemoryError: If appending would exceed window size
        """
        # Combine with any pending bytes from previous chunk
        if self._pending_bytes:
            data = bytes(self._pending_bytes) + data
            self._pending_bytes.clear()

        # Check if incoming data is larger than max_size
        if len(data) > self.max_size:
            raise MemoryError(f"Incoming data size {len(data)} exceeds maximum window size {self.max_size}")

        # Check memory constraint
        if len(self.buffer) + len(data) > self.max_size:
            # Need to slide window first
            if not self.can_slide():
                logger.error(
                    f"Window size would exceed {self.max_size} bytes. "
                    f"Current buffer: {len(self.buffer)}, incoming: {len(data)}"
                )
                raise MemoryError(
                    f"Window size would exceed {self.max_size} bytes. "
                    "Process or slide window before appending more data."
                )

            # Calculate how much we need to slide
            needed_space = len(data)
            available_after_slide = self.max_size - (len(self.buffer) // 2)  # After default slide

            if needed_space > available_after_slide:
                # Even after sliding, we won't have enough space
                logger.error(
                    f"Cannot fit {len(data)} bytes even after sliding. "
                    f"Max size: {self.max_size}, would have after slide: {available_after_slide}"
                )
                raise MemoryError(
                    f"Cannot fit {len(data)} bytes even after sliding window. "
                    f"Maximum available: {available_after_slide} bytes"
                )

            logger.debug(f"Sliding window to make room for {len(data)} bytes")
            self.slide()

        self.buffer.extend(data)
        logger.debug(f"Appended {len(data)} bytes to window, new size: {len(self.buffer)}")

    def decode_safe(self) -> str:
        """
        Safely decode buffer to text, handling UTF-8 boundaries.

        Returns:
            Decoded text string
        """
        if not self.buffer:
            return ""

        # Find safe UTF-8 boundary
        safe_end = self._find_utf8_boundary(self.buffer)

        # If safe_end is 0, the entire buffer is incomplete or invalid UTF-8
        if safe_end == 0:
            # Store entire buffer as pending and return empty string
            if not self._pending_bytes:
                self._pending_bytes = bytearray()
            self._pending_bytes.extend(self.buffer)
            self.buffer = bytearray()
            return ""

        # Split at boundary
        decodable = self.buffer[:safe_end]

        # Store incomplete bytes for next iteration
        if safe_end < len(self.buffer):
            if not self._pending_bytes:
                self._pending_bytes = bytearray()
            self._pending_bytes.extend(self.buffer[safe_end:])

        # Clear the main buffer after extracting decodable portion
        self.buffer = bytearray()

        # Decode the safe portion
        try:
            text = decodable.decode("utf-8", errors="strict")
            self.text_buffer = text
            return text
        except UnicodeDecodeError as e:
            # This shouldn't happen if boundary detection is correct
            # Log the error and raise exception instead of silently masking
            logger.error(
                f"UTF-8 decode error at position {e.start}-{e.end}: {e.reason}. "
                f"This indicates a bug in UTF-8 boundary detection. "
                f"Buffer size: {len(decodable)}, Safe end: {safe_end}"
            )
            raise ValueError(
                f"Failed to decode UTF-8 text: {e.reason}. "
                "This indicates a bug in the UTF-8 boundary detection logic."
            ) from e

    def _find_utf8_boundary(self, data: bytes, from_end: bool = True) -> int:  # noqa: ARG002
        """
        Find a safe UTF-8 character boundary in the data.

        Args:
            data: Byte data to search
            from_end: If True, search from end; else from start

        Returns:
            Safe boundary position
        """
        if not data:
            return 0

        # Always validate from the start to find the last valid boundary
        pos = 0
        last_valid_boundary = 0

        while pos < len(data):
            byte = data[pos]

            if byte < 0x80:
                # ASCII character (0xxxxxxx)
                pos += 1
                last_valid_boundary = pos
            elif byte < 0xC0:
                # Continuation byte (10xxxxxx) at start position - invalid
                # No valid UTF-8 from this point
                break
            elif byte < 0xE0:
                # 2-byte sequence (110xxxxx)
                if pos + 2 > len(data):
                    # Incomplete sequence
                    break
                # Check continuation byte
                if not (0x80 <= data[pos + 1] < 0xC0):
                    # Invalid continuation
                    break
                pos += 2
                last_valid_boundary = pos
            elif byte < 0xF0:
                # 3-byte sequence (1110xxxx)
                if pos + 3 > len(data):
                    # Incomplete sequence
                    break
                # Check continuation bytes
                if not all(0x80 <= data[pos + i] < 0xC0 for i in range(1, 3)):
                    # Invalid continuation
                    break
                pos += 3
                last_valid_boundary = pos
            elif byte < 0xF8:
                # 4-byte sequence (11110xxx)
                if pos + 4 > len(data):
                    # Incomplete sequence
                    break
                # Check continuation bytes
                if not all(0x80 <= data[pos + i] < 0xC0 for i in range(1, 4)):
                    # Invalid continuation
                    break
                pos += 4
                last_valid_boundary = pos
            else:
                # Invalid UTF-8 start byte
                break

        return last_valid_boundary

    def _get_utf8_char_length(self, first_byte: int) -> int:
        """
        Get the expected length of a UTF-8 character from its first byte.

        Args:
            first_byte: First byte of UTF-8 sequence

        Returns:
            Expected character length in bytes
        """
        if first_byte < 0x80:
            return 1  # ASCII
        if first_byte < 0xE0:
            return 2  # 110xxxxx
        if first_byte < 0xF0:
            return 3  # 1110xxxx
        if first_byte < 0xF8:
            return 4  # 11110xxx
        # Invalid UTF-8 start byte
        return 1

    def is_ready(self) -> bool:
        """
        Check if window has enough data for processing.

        Returns:
            True if window is ready for processing
        """
        # Ready if we have at least 80% of max size or any data with EOF
        return len(self.buffer) >= int(self.max_size * 0.8)

    def can_slide(self) -> bool:
        """
        Check if window can slide forward.

        Returns:
            True if window has processable content
        """
        return len(self.buffer) > 0

    def slide(self, amount: int | None = None) -> bytes:
        """
        Slide the window forward, returning processed bytes.

        Args:
            amount: Bytes to slide (default: half the buffer)

        Returns:
            Bytes that were removed from the window
        """
        if not self.buffer:
            return b""

        if amount is None:
            # Default: slide by half the buffer size
            amount = len(self.buffer) // 2

        # Ensure we slide at a UTF-8 boundary
        safe_amount = self._find_utf8_boundary(self.buffer[:amount])

        # Extract the sliding portion
        sliding_data = bytes(self.buffer[:safe_amount])

        # Remove from buffer
        self.buffer = self.buffer[safe_amount:]

        # Update offsets
        self.byte_offset += safe_amount

        logger.debug(
            f"Slid window by {safe_amount} bytes (requested: {amount}), "
            f"new offset: {self.byte_offset}, remaining: {len(self.buffer)}"
        )

        return sliding_data

    def get_overlap_zone(self, overlap_size: int) -> bytes:
        """
        Get the overlap zone for the next window.

        Args:
            overlap_size: Size of overlap in bytes

        Returns:
            Bytes from the end of current window for overlap
        """
        if not self.buffer or overlap_size <= 0:
            return b""

        # Ensure overlap doesn't exceed buffer
        actual_overlap = min(overlap_size, len(self.buffer))

        # Find safe UTF-8 boundary for overlap
        overlap_start = len(self.buffer) - actual_overlap
        safe_start = overlap_start

        # Adjust to character boundary
        while safe_start > 0:
            if self.buffer[safe_start] < 0x80 or self.buffer[safe_start] >= 0xC0:
                break
            safe_start -= 1

        return bytes(self.buffer[safe_start:])

    def clear(self) -> None:
        """Clear the window buffer."""
        self.buffer.clear()
        self.text_buffer = ""
        self._pending_bytes.clear()

    def remaining_capacity(self) -> int:
        """
        Get remaining capacity in bytes.

        Returns:
            Number of bytes that can be added before hitting limit
        """
        return self.max_size - len(self.buffer)

    @property
    def size(self) -> int:
        """Get current buffer size in bytes."""
        return len(self.buffer)

    @property
    def is_empty(self) -> bool:
        """Check if window is empty."""
        return len(self.buffer) == 0

    def __repr__(self) -> str:
        """String representation of the window."""
        return (
            f"StreamingWindow(size={len(self.buffer)}/{self.max_size}, "
            f"offset={self.byte_offset}, pending={len(self._pending_bytes)})"
        )
