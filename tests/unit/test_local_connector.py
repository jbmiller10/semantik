"""Unit tests for LocalFileConnector."""

from pathlib import Path

import pytest

from shared.connectors.local import LocalFileConnector


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


class TestLocalFileConnectorEnumerate:
    """Tests for enumerate method."""

    @pytest.mark.asyncio()
    async def test_enumerate_txt_file(self, tmp_path: Path) -> None:
        """Test enumerating a simple text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")

        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 1
        assert refs[0].uri == f"file://{txt_file}"
        assert refs[0].source_type == "directory"
        assert refs[0].content_type == "document"
        assert refs[0].filename == "test.txt"
        assert refs[0].extension == ".txt"
        assert refs[0].mime_type == "text/plain"
        assert refs[0].size_bytes == len("Hello, world!")
        assert refs[0].change_hint is not None
        assert refs[0].change_hint.startswith("mtime:")
        assert ",size:" in refs[0].change_hint
        assert refs[0].source_metadata["local_path"] == str(txt_file)
        assert refs[0].source_metadata["relative_path"] == "test.txt"

    @pytest.mark.asyncio()
    async def test_enumerate_md_file(self, tmp_path: Path) -> None:
        """Test enumerating a markdown file."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Heading\n\nSome content.")

        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 1
        assert refs[0].filename == "readme.md"
        assert refs[0].extension == ".md"
        assert refs[0].mime_type == "text/markdown"

    @pytest.mark.asyncio()
    async def test_enumerate_recursive(self, tmp_path: Path) -> None:
        """Test recursive directory traversal."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root content")
        (subdir / "nested.txt").write_text("nested content")

        connector = LocalFileConnector({"path": str(tmp_path), "recursive": True})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 2
        filenames = [ref.filename for ref in refs]
        assert "root.txt" in filenames
        assert "nested.txt" in filenames

    @pytest.mark.asyncio()
    async def test_enumerate_non_recursive(self, tmp_path: Path) -> None:
        """Test non-recursive scanning."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root content")
        (subdir / "nested.txt").write_text("nested content")

        connector = LocalFileConnector({"path": str(tmp_path), "recursive": False})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 1
        assert refs[0].filename == "root.txt"

    @pytest.mark.asyncio()
    async def test_enumerate_empty_directory(self, tmp_path: Path) -> None:
        """Test enumerating from empty directory."""
        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]
        assert len(refs) == 0

    @pytest.mark.asyncio()
    async def test_enumerate_multiple_files(self, tmp_path: Path) -> None:
        """Test enumerating multiple files."""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "file3.md").write_text("# Content 3")

        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 3

    @pytest.mark.asyncio()
    async def test_enumerate_change_hint_format(self, tmp_path: Path) -> None:
        """Test change_hint contains mtime and size."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!")

        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 1
        # change_hint format: mtime:{timestamp},size:{bytes}
        change_hint = refs[0].change_hint
        assert change_hint is not None
        assert change_hint.startswith("mtime:")
        assert ",size:" in change_hint

        # Parse and verify
        parts = change_hint.split(",")
        mtime_part = parts[0]
        size_part = parts[1]
        assert mtime_part.startswith("mtime:")
        assert size_part.startswith("size:")
        assert int(mtime_part.split(":")[1]) > 0
        assert int(size_part.split(":")[1]) == len("Hello, world!")

    @pytest.mark.asyncio()
    async def test_enumerate_source_metadata(self, tmp_path: Path) -> None:
        """Test source_metadata contains local_path and relative_path."""
        subdir = tmp_path / "docs"
        subdir.mkdir()
        txt_file = subdir / "test.txt"
        txt_file.write_text("content")

        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 1
        assert refs[0].source_metadata["local_path"] == str(txt_file)
        assert refs[0].source_metadata["relative_path"] == "docs/test.txt"

    @pytest.mark.asyncio()
    async def test_enumerate_with_include_patterns(self, tmp_path: Path) -> None:
        """Test include_patterns filtering."""
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "notes.txt").write_text("Notes")
        (tmp_path / "data.json").write_text("{}")

        connector = LocalFileConnector(
            {
                "path": str(tmp_path),
                "include_patterns": ["*.md"],
            }
        )
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 1
        assert refs[0].filename == "notes.md"

    @pytest.mark.asyncio()
    async def test_enumerate_with_exclude_patterns(self, tmp_path: Path) -> None:
        """Test exclude_patterns filtering."""
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "secret.md").write_text("# Secret")
        (tmp_path / "data.txt").write_text("data")

        connector = LocalFileConnector(
            {
                "path": str(tmp_path),
                "exclude_patterns": ["secret*"],
            }
        )
        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 2
        filenames = [ref.filename for ref in refs]
        assert "notes.md" in filenames
        assert "data.txt" in filenames
        assert "secret.md" not in filenames

    @pytest.mark.asyncio()
    async def test_enumerate_infers_content_type(self, tmp_path: Path) -> None:
        """Test content_type inference based on extension."""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "code.py").write_text("print('hello')")
        (tmp_path / "readme.md").write_text("# README")

        connector = LocalFileConnector({"path": str(tmp_path)})
        refs = [ref async for ref in connector.enumerate()]

        ref_by_name = {ref.filename: ref for ref in refs}
        assert ref_by_name["doc.pdf"].content_type == "document"
        assert ref_by_name["code.py"].content_type == "code"
        assert ref_by_name["readme.md"].content_type == "document"


class TestLocalFileConnectorConfigFields:
    """Tests for config field methods."""

    def test_get_config_fields_includes_path(self) -> None:
        """Test get_config_fields includes path field."""
        fields = LocalFileConnector.get_config_fields()
        field_names = [f["name"] for f in fields]
        assert "path" in field_names

    def test_get_config_fields_includes_recursive(self) -> None:
        """Test get_config_fields includes recursive field."""
        fields = LocalFileConnector.get_config_fields()
        field_names = [f["name"] for f in fields]
        assert "recursive" in field_names

    def test_get_secret_fields_is_empty(self) -> None:
        """Test get_secret_fields returns empty list."""
        fields = LocalFileConnector.get_secret_fields()
        assert fields == []
