"""Unit tests for content hashing utilities."""

from shared.utils.hashing import compute_content_hash


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_simple_ascii_content(self) -> None:
        """Test hashing simple ASCII content."""
        result = compute_content_hash("Hello, world!")
        assert result == "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"

    def test_returns_64_character_hex_string(self) -> None:
        """Test that result is always 64 lowercase hex characters."""
        result = compute_content_hash("test content")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string(self) -> None:
        """Test hashing empty string."""
        result = compute_content_hash("")
        # SHA-256 of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_multibyte_utf8_characters(self) -> None:
        """Test hashing content with multi-byte UTF-8 characters."""
        # Emoji
        result_emoji = compute_content_hash("Hello 🌍 World!")
        assert len(result_emoji) == 64

        # CJK characters
        result_cjk = compute_content_hash("你好世界")
        assert len(result_cjk) == 64

        # Mixed content
        result_mixed = compute_content_hash("日本語 🎉 Français ñ")
        assert len(result_mixed) == 64

    def test_bytes_input(self) -> None:
        """Test hashing binary content."""
        result = compute_content_hash(b"Hello, world!")
        # Same as string version encoded to UTF-8
        assert result == "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"

    def test_binary_data(self) -> None:
        """Test hashing arbitrary binary data."""
        binary_data = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD])
        result = compute_content_hash(binary_data)
        assert len(result) == 64

    def test_determinism(self) -> None:
        """Test that same input always produces same output."""
        content = "deterministic test content"
        result1 = compute_content_hash(content)
        result2 = compute_content_hash(content)
        result3 = compute_content_hash(content)
        assert result1 == result2 == result3

    def test_different_content_different_hash(self) -> None:
        """Test that different content produces different hashes."""
        hash1 = compute_content_hash("content A")
        hash2 = compute_content_hash("content B")
        assert hash1 != hash2

    def test_whitespace_sensitive(self) -> None:
        """Test that whitespace differences produce different hashes."""
        hash1 = compute_content_hash("hello world")
        hash2 = compute_content_hash("hello  world")  # Extra space
        hash3 = compute_content_hash("hello world ")  # Trailing space
        hash4 = compute_content_hash(" hello world")  # Leading space

        # All should be different
        hashes = [hash1, hash2, hash3, hash4]
        assert len(set(hashes)) == 4

    def test_newline_handling(self) -> None:
        """Test that newlines are included in hash."""
        hash_no_newline = compute_content_hash("line1line2")
        hash_with_lf = compute_content_hash("line1\nline2")
        hash_with_crlf = compute_content_hash("line1\r\nline2")

        # All should be different
        assert hash_no_newline != hash_with_lf
        assert hash_with_lf != hash_with_crlf
