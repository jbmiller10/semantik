"""
Tests for document extraction module
"""

import os
import tempfile
import pytest

from vecpipe.extract_chunks_v2 import (
    TokenChunker,
    FileChangeTracker,
    extract_text_txt,
    extract_text_pdf,
    extract_text_docx,
)


class TestTokenChunker:
    """Test token-based chunking"""

    def test_empty_text(self):
        """Test chunking empty text"""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text("", "doc1")
        assert len(chunks) == 0

    def test_small_text(self):
        """Test text smaller than chunk size"""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a small test text."
        chunks = chunker.chunk_text(text, "doc1")

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["doc_id"] == "doc1"
        assert chunks[0]["chunk_id"] == "doc1_0000"

    def test_multi_chunk_text(self):
        """Test text requiring multiple chunks"""
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        # Create text that will require multiple chunks
        text = " ".join(["word"] * 200)  # ~200 tokens
        chunks = chunker.chunk_text(text, "doc1")

        assert len(chunks) > 1
        assert all(c["doc_id"] == "doc1" for c in chunks)
        assert all(c["token_count"] <= 50 for c in chunks)

        # Check chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"doc1_{i:04d}"

    def test_sentence_boundary_breaking(self):
        """Test that chunker respects sentence boundaries"""
        chunker = TokenChunker(chunk_size=20, chunk_overlap=5)
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk_text(text, "doc1")

        # Should break at sentence boundaries when possible
        for chunk in chunks[:-1]:  # All but last chunk
            assert chunk["text"].rstrip().endswith(".")


class TestFileChangeTracker:
    """Test file change tracking"""

    def test_new_file(self):
        """Test tracking a new file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileChangeTracker(os.path.join(tmpdir, "tracking.json"))

            # Create a test file
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            should_process, file_hash = tracker.should_process_file(test_file)

            assert should_process is True
            assert file_hash is not None

    def test_unchanged_file(self):
        """Test that unchanged files are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileChangeTracker(os.path.join(tmpdir, "tracking.json"))

            # Create and track a file
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            should_process1, hash1 = tracker.should_process_file(test_file)
            tracker.update_file_tracking(test_file, hash1, "doc1", 5)

            # Check same file again
            should_process2, hash2 = tracker.should_process_file(test_file)

            assert should_process1 is True
            assert should_process2 is False
            assert hash1 == hash2

    def test_changed_file(self):
        """Test that changed files are reprocessed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileChangeTracker(os.path.join(tmpdir, "tracking.json"))

            # Create and track a file
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("original content")

            should_process1, hash1 = tracker.should_process_file(test_file)
            tracker.update_file_tracking(test_file, hash1, "doc1", 5)

            # Modify the file
            with open(test_file, "w") as f:
                f.write("modified content")

            should_process2, hash2 = tracker.should_process_file(test_file)

            assert should_process1 is True
            assert should_process2 is True
            assert hash1 != hash2

    def test_removed_files_detection(self):
        """Test detection of removed files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileChangeTracker(os.path.join(tmpdir, "tracking.json"))

            # Track some files
            files = []
            for i in range(3):
                test_file = os.path.join(tmpdir, f"test{i}.txt")
                with open(test_file, "w") as f:
                    f.write(f"content {i}")
                files.append(test_file)

                _, file_hash = tracker.should_process_file(test_file)
                tracker.update_file_tracking(test_file, file_hash, f"doc{i}", 5)

            # Remove one file from current list
            current_files = files[:2]  # Simulate file 2 being removed

            removed = tracker.get_removed_files(current_files)

            assert len(removed) == 1
            assert removed[0]["doc_id"] == "doc2"


class TestTextExtraction:
    """Test text extraction from different file types"""

    def test_extract_txt(self):
        """Test TXT file extraction"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test file.\nWith multiple lines.")
            f.flush()

            text = extract_text_txt(f.name)

            assert "This is a test file." in text
            assert "With multiple lines." in text

        os.unlink(f.name)

    def test_extract_txt_with_encoding_issues(self):
        """Test TXT extraction with encoding issues"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            # Write some bytes that might cause encoding issues
            f.write(b"Normal text \xc0\xc1 more text")
            f.flush()

            # Should not raise exception
            text = extract_text_txt(f.name)
            assert "Normal text" in text
            assert "more text" in text

        os.unlink(f.name)


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_long_chunk(self):
        """Test handling of very long chunks"""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        # Create a very long "word" that can't be broken
        long_word = "a" * 1000
        chunks = chunker.chunk_text(long_word, "doc1")

        # Should still produce chunks, even if they exceed size limit
        assert len(chunks) >= 1

    def test_unicode_text(self):
        """Test handling of unicode text"""
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        text = "Hello 世界! This is a test with émojis 🎉 and special çharacters."
        chunks = chunker.chunk_text(text, "doc1")

        assert len(chunks) >= 1
        # Verify text is preserved correctly
        combined = " ".join(c["text"] for c in chunks)
        assert "世界" in combined
        assert "🎉" in combined

    @pytest.mark.parametrize(
        "chunk_size,overlap",
        [
            (100, 20),
            (500, 100),
            (1000, 200),
        ],
    )
    def test_different_chunk_sizes(self, chunk_size, overlap):
        """Test various chunk size configurations"""
        chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        text = " ".join(["word"] * 1000)  # ~1000 tokens
        chunks = chunker.chunk_text(text, "doc1")

        assert len(chunks) > 0
        # Verify no chunk exceeds the size limit (except possibly the last)
        for chunk in chunks[:-1]:
            assert chunk["token_count"] <= chunk_size
