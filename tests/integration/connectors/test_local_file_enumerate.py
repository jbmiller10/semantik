"""Integration tests for LocalFileConnector enumerate functionality.

These tests use the real filesystem without mocking to verify enumerate
behavior in realistic scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from shared.connectors.local import LocalFileConnector

if TYPE_CHECKING:
    from pathlib import Path


class TestLocalFileConnectorEnumerateIntegration:
    """Integration tests for LocalFileConnector.enumerate()."""

    @pytest.mark.asyncio()
    async def test_enumerate_yields_correct_file_references(self, tmp_path: Path) -> None:
        """Test enumerate yields FileReference objects with correct fields."""
        # Create test files
        doc_file = tmp_path / "document.txt"
        doc_file.write_text("Hello, world!")

        code_file = tmp_path / "script.py"
        code_file.write_text("print('hello')")

        connector = LocalFileConnector({"path": str(tmp_path)})

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 2

        # Check each FileReference has required fields
        for ref in refs:
            assert ref.uri is not None
            assert ref.source_type == "directory"
            assert ref.content_type in ("document", "code")
            assert ref.filename is not None
            assert ref.extension in (".txt", ".py")
            assert ref.size_bytes > 0
            assert ref.change_hint is not None
            assert ref.source_metadata is not None
            assert "local_path" in ref.source_metadata
            assert "relative_path" in ref.source_metadata

    @pytest.mark.asyncio()
    async def test_enumerate_handles_nested_directories(self, tmp_path: Path) -> None:
        """Test enumerate correctly handles nested directory structures."""
        # Create nested structure
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "level3").mkdir()

        (tmp_path / "root.txt").write_text("root")
        (tmp_path / "level1" / "l1.txt").write_text("level1")
        (tmp_path / "level1" / "level2" / "l2.txt").write_text("level2")
        (tmp_path / "level1" / "level2" / "level3" / "l3.txt").write_text("level3")

        connector = LocalFileConnector({"path": str(tmp_path), "recursive": True})

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 4
        relative_paths = {ref.source_metadata["relative_path"] for ref in refs}
        assert "root.txt" in relative_paths
        assert "level1/l1.txt" in relative_paths
        assert "level1/level2/l2.txt" in relative_paths
        assert "level1/level2/level3/l3.txt" in relative_paths

    @pytest.mark.asyncio()
    async def test_enumerate_symlink_security(self, tmp_path: Path) -> None:
        """Test enumerate blocks symlinks pointing outside base directory."""
        # Create a file outside the source directory
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret data")

        # Create source directory with symlink to outside
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "legit.txt").write_text("legitimate file")

        # Create symlink pointing outside source
        symlink = source_dir / "link_to_secret.txt"
        symlink.symlink_to(outside_file)

        connector = LocalFileConnector({"path": str(source_dir)})

        refs = [ref async for ref in connector.enumerate()]

        # Should only enumerate legit.txt, not the symlink
        assert len(refs) == 1
        assert refs[0].filename == "legit.txt"

    @pytest.mark.asyncio()
    async def test_enumerate_change_hint_changes_with_content(self, tmp_path: Path) -> None:
        """Test that change_hint changes when file content changes."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("original content")

        connector = LocalFileConnector({"path": str(tmp_path)})

        # First enumeration
        refs1 = [ref async for ref in connector.enumerate()]
        change_hint_1 = refs1[0].change_hint

        # Modify the file - use longer sleep to ensure mtime changes
        # (some filesystems have 1-second resolution)
        import time

        time.sleep(1.1)
        file_path.write_text("modified content!!")  # Different content length too

        # Second enumeration
        refs2 = [ref async for ref in connector.enumerate()]
        change_hint_2 = refs2[0].change_hint

        # Change hints should be different (either mtime or size changed)
        assert change_hint_1 != change_hint_2

    @pytest.mark.asyncio()
    async def test_enumerate_include_patterns(self, tmp_path: Path) -> None:
        """Test include_patterns filtering works correctly."""
        # Create various file types
        (tmp_path / "doc.md").write_text("# Document")
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "script.py").write_text("print('hi')")
        (tmp_path / "data.json").write_text("{}")

        connector = LocalFileConnector(
            {
                "path": str(tmp_path),
                "include_patterns": ["*.md"],
            }
        )

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 2
        filenames = {ref.filename for ref in refs}
        assert filenames == {"doc.md", "readme.md"}

    @pytest.mark.asyncio()
    async def test_enumerate_exclude_patterns(self, tmp_path: Path) -> None:
        """Test exclude_patterns filtering works correctly."""
        # Create various files
        (tmp_path / "app.py").write_text("# App")
        (tmp_path / "app.pyc").write_bytes(b"\x00")
        (tmp_path / "test.py").write_text("# Test")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.pyc").write_bytes(b"\x00")

        connector = LocalFileConnector(
            {
                "path": str(tmp_path),
                "exclude_patterns": ["*.pyc", "__pycache__/*"],
            }
        )

        refs = [ref async for ref in connector.enumerate()]

        filenames = {ref.filename for ref in refs}
        assert "app.py" in filenames
        assert "test.py" in filenames
        assert "app.pyc" not in filenames
        assert "cache.pyc" not in filenames

    @pytest.mark.asyncio()
    async def test_enumerate_empty_directory(self, tmp_path: Path) -> None:
        """Test enumerate handles empty directories gracefully."""
        connector = LocalFileConnector({"path": str(tmp_path)})

        refs = [ref async for ref in connector.enumerate()]

        assert refs == []

    @pytest.mark.asyncio()
    async def test_enumerate_large_directory(self, tmp_path: Path) -> None:
        """Test enumerate handles directories with many files."""
        # Create 100 files
        for i in range(100):
            (tmp_path / f"file_{i:03d}.txt").write_text(f"Content {i}")

        connector = LocalFileConnector({"path": str(tmp_path)})

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == 100
        # Verify all files are unique
        uris = {ref.uri for ref in refs}
        assert len(uris) == 100

    @pytest.mark.asyncio()
    async def test_enumerate_various_extensions(self, tmp_path: Path) -> None:
        """Test enumerate correctly identifies various file types."""
        # Create files with various extensions
        # Note: json, yaml, etc. are classified as "config" not "document"
        extensions = {
            ".txt": "document",
            ".md": "document",
            ".pdf": "document",
            ".py": "code",
            ".js": "code",
            ".ts": "code",
            ".java": "code",
            ".rs": "code",
            ".html": "document",
            ".json": "config",
        }

        for ext, _ in extensions.items():
            filename = f"test{ext}"
            if ext == ".pdf":
                (tmp_path / filename).write_bytes(b"%PDF-1.4")
            else:
                (tmp_path / filename).write_text(f"content for {ext}")

        connector = LocalFileConnector({"path": str(tmp_path)})

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == len(extensions)

        # Verify content_type inference
        for ref in refs:
            ext = ref.extension
            expected_type = extensions.get(ext, "document") if ext else "document"
            assert ref.content_type == expected_type, f"Expected {expected_type} for {ext}, got {ref.content_type}"

    @pytest.mark.asyncio()
    async def test_enumerate_special_characters_in_filenames(self, tmp_path: Path) -> None:
        """Test enumerate handles filenames with special characters."""
        # Create files with special characters (filesystem-safe ones)
        special_names = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.multiple.dots.txt",
        ]

        for name in special_names:
            (tmp_path / name).write_text("content")

        connector = LocalFileConnector({"path": str(tmp_path)})

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == len(special_names)
        filenames = {ref.filename for ref in refs}
        for name in special_names:
            assert name in filenames

    @pytest.mark.asyncio()
    async def test_enumerate_unicode_filenames(self, tmp_path: Path) -> None:
        """Test enumerate handles unicode filenames."""
        unicode_names = [
            "文档.txt",
            "документ.txt",
            "日本語.txt",
        ]

        for name in unicode_names:
            try:
                (tmp_path / name).write_text("content")
            except OSError:
                # Skip if filesystem doesn't support unicode
                pytest.skip("Filesystem doesn't support unicode filenames")

        connector = LocalFileConnector({"path": str(tmp_path)})

        refs = [ref async for ref in connector.enumerate()]

        assert len(refs) == len(unicode_names)

    @pytest.mark.asyncio()
    async def test_enumerate_lazy_iteration(self, tmp_path: Path) -> None:
        """Test that enumerate uses lazy iteration (doesn't load all at once)."""
        # Create files
        for i in range(10):
            (tmp_path / f"file_{i}.txt").write_text(f"Content {i}")

        connector = LocalFileConnector({"path": str(tmp_path)})

        # Get the async generator
        gen = connector.enumerate()

        # Consume only first 3 items
        count = 0
        async for _ in gen:
            count += 1
            if count >= 3:
                break

        assert count == 3
        # Generator should still be usable (lazy iteration)
