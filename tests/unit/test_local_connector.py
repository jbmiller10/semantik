"""Unit tests for LocalFileConnector."""

from pathlib import Path

import pytest

from shared.connectors.local import (
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
    LocalFileConnector,
)


class TestLocalFileConnectorConfig:
    """Tests for config validation."""

    def test_validate_config_requires_path(self) -> None:
        """Test that path is required in config."""
        with pytest.raises(ValueError, match="requires 'path'"):
            LocalFileConnector({})

    def test_validate_config_with_path(self) -> None:
        """Test valid config with path."""
        connector = LocalFileConnector({"path": "/tmp"})
        assert connector.config["path"] == "/tmp"

    def test_config_defaults_recursive_true(self) -> None:
        """Test recursive defaults to True."""
        connector = LocalFileConnector({"path": "/tmp"})
        assert connector.config.get("recursive", True) is True

    def test_config_can_set_recursive_false(self) -> None:
        """Test recursive can be explicitly set to False."""
        connector = LocalFileConnector({"path": "/tmp", "recursive": False})
        assert connector.config["recursive"] is False


class TestLocalFileConnectorAuthenticate:
    """Tests for authenticate method."""

    @pytest.mark.asyncio()
    async def test_authenticate_existing_directory(self, tmp_path: Path) -> None:
        """Test authenticate passes for existing directory."""
        connector = LocalFileConnector({"path": str(tmp_path)})
        assert await connector.authenticate() is True

    @pytest.mark.asyncio()
    async def test_authenticate_nonexistent_path(self) -> None:
        """Test authenticate raises for nonexistent path."""
        connector = LocalFileConnector({"path": "/nonexistent/path/xyz123"})
        with pytest.raises(ValueError, match="does not exist"):
            await connector.authenticate()

    @pytest.mark.asyncio()
    async def test_authenticate_file_not_directory(self, tmp_path: Path) -> None:
        """Test authenticate raises for file path."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        connector = LocalFileConnector({"path": str(file_path)})
        with pytest.raises(ValueError, match="not a directory"):
            await connector.authenticate()


class TestLocalFileConnectorLoadDocuments:
    """Tests for load_documents method."""

    @pytest.mark.asyncio()
    async def test_load_documents_txt_file(self, tmp_path: Path) -> None:
        """Test loading a simple text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")

        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 1
        assert "Hello" in docs[0].content
        assert docs[0].source_type == "directory"
        assert docs[0].unique_id == f"file://{txt_file}"
        assert len(docs[0].content_hash) == 64
        assert docs[0].file_path == str(txt_file)

    @pytest.mark.asyncio()
    async def test_load_documents_md_file(self, tmp_path: Path) -> None:
        """Test loading a markdown file."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Heading\n\nSome content.")

        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 1
        assert "Heading" in docs[0].content
        assert docs[0].metadata.get("mime_type") == "text/markdown"

    @pytest.mark.asyncio()
    async def test_load_documents_skips_unsupported(self, tmp_path: Path) -> None:
        """Test that unsupported extensions are skipped."""
        (tmp_path / "test.xyz").write_text("content")
        (tmp_path / "test.txt").write_text("valid")

        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 1
        assert docs[0].file_path is not None
        assert docs[0].file_path.endswith(".txt")

    @pytest.mark.asyncio()
    async def test_load_documents_recursive(self, tmp_path: Path) -> None:
        """Test recursive directory traversal."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root content")
        (subdir / "nested.txt").write_text("nested content")

        connector = LocalFileConnector({"path": str(tmp_path), "recursive": True})
        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 2
        file_paths = [doc.file_path for doc in docs]
        assert any("root.txt" in fp for fp in file_paths if fp)
        assert any("nested.txt" in fp for fp in file_paths if fp)

    @pytest.mark.asyncio()
    async def test_load_documents_non_recursive(self, tmp_path: Path) -> None:
        """Test non-recursive scanning."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root content")
        (subdir / "nested.txt").write_text("nested content")

        connector = LocalFileConnector({"path": str(tmp_path), "recursive": False})
        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 1
        assert "root" in docs[0].content

    @pytest.mark.asyncio()
    async def test_load_documents_empty_directory(self, tmp_path: Path) -> None:
        """Test loading from empty directory."""
        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]
        assert len(docs) == 0

    @pytest.mark.asyncio()
    async def test_load_documents_multiple_files(self, tmp_path: Path) -> None:
        """Test loading multiple files."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "file3.md").write_text("# Content 3")

        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 3

    @pytest.mark.asyncio()
    async def test_load_documents_content_hash_is_deterministic(self, tmp_path: Path) -> None:
        """Test content hash is deterministic for same content."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")

        connector1 = LocalFileConnector({"path": str(tmp_path)})
        docs1 = [doc async for doc in connector1.load_documents()]

        connector2 = LocalFileConnector({"path": str(tmp_path)})
        docs2 = [doc async for doc in connector2.load_documents()]

        assert docs1[0].content_hash == docs2[0].content_hash

    @pytest.mark.asyncio()
    async def test_load_documents_different_content_different_hash(self, tmp_path: Path) -> None:
        """Test different content produces different hashes."""
        (tmp_path / "file1.txt").write_text("Content A")
        (tmp_path / "file2.txt").write_text("Content B")

        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]

        assert docs[0].content_hash != docs[1].content_hash

    @pytest.mark.asyncio()
    async def test_load_documents_metadata_includes_file_size(self, tmp_path: Path) -> None:
        """Test metadata includes file size."""
        content = "Hello, world!"
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(content)

        connector = LocalFileConnector({"path": str(tmp_path)})
        docs = [doc async for doc in connector.load_documents()]

        assert "file_size" in docs[0].metadata
        assert docs[0].metadata["file_size"] > 0


class TestLocalFileConnectorConstants:
    """Tests for module constants."""

    def test_supported_extensions_includes_common_types(self) -> None:
        """Test SUPPORTED_EXTENSIONS includes common document types."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".html" in SUPPORTED_EXTENSIONS

    def test_max_file_size_is_500mb(self) -> None:
        """Test MAX_FILE_SIZE is 500 MB."""
        assert MAX_FILE_SIZE == 500 * 1024 * 1024
