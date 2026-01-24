"""Unit tests for source analysis sub-agent tools.

These tests verify the tools used by the SourceAnalyzer sub-agent
for investigating data sources.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.pipeline.types import FileReference
from webui.services.agent.tools.subagent_tools.source import (
    DetectLanguageTool,
    EnumerateFilesTool,
    GetFileContentPreviewTool,
    SampleFilesTool,
    TryParserTool,
)


# Fixtures
@pytest.fixture
def sample_file_refs() -> list[FileReference]:
    """Create sample file references for testing."""
    return [
        FileReference(
            uri="file:///docs/paper1.pdf",
            source_type="local_directory",
            content_type="file",
            filename="paper1.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=100000,
            change_hint="mtime:1234567890,size:100000",
            source_metadata={"local_path": "/docs/paper1.pdf"},
        ),
        FileReference(
            uri="file:///docs/paper2.pdf",
            source_type="local_directory",
            content_type="file",
            filename="paper2.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=200000,
            change_hint="mtime:1234567891,size:200000",
            source_metadata={"local_path": "/docs/paper2.pdf"},
        ),
        FileReference(
            uri="file:///docs/notes.md",
            source_type="local_directory",
            content_type="file",
            filename="notes.md",
            extension=".md",
            mime_type="text/markdown",
            size_bytes=5000,
            change_hint="mtime:1234567892,size:5000",
            source_metadata={"local_path": "/docs/notes.md"},
        ),
        FileReference(
            uri="file:///docs/code.py",
            source_type="local_directory",
            content_type="file",
            filename="code.py",
            extension=".py",
            mime_type="text/x-python",
            size_bytes=2000,
            change_hint="mtime:1234567893,size:2000",
            source_metadata={"local_path": "/docs/code.py"},
        ),
    ]


@pytest.fixture
def mock_connector(sample_file_refs: list[FileReference]) -> AsyncMock:
    """Create a mock connector that yields sample file refs."""
    connector = AsyncMock()
    connector.authenticate = AsyncMock(return_value=True)

    async def mock_enumerate(source_id: int | None = None):
        for ref in sample_file_refs:
            yield ref

    connector.enumerate = mock_enumerate
    connector.load_content = AsyncMock(return_value=b"Sample content for testing")
    return connector


# EnumerateFilesTool tests
class TestEnumerateFilesTool:
    """Tests for the EnumerateFilesTool."""

    @pytest.mark.asyncio
    async def test_enumerate_returns_statistics(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test that enumeration returns correct statistics."""
        context = {"connector": mock_connector, "source_id": 1}
        tool = EnumerateFilesTool(context)

        result = await tool.execute(include_samples=True, samples_per_type=2)

        assert result["success"] is True
        assert result["total_files"] == 4
        assert result["extension_count"] == 3  # .pdf, .md, .py
        assert ".pdf" in result["by_extension"]
        assert result["by_extension"][".pdf"]["count"] == 2

    @pytest.mark.asyncio
    async def test_enumerate_includes_samples(self, mock_connector: AsyncMock):
        """Test that samples are included when requested."""
        context = {"connector": mock_connector, "source_id": 1}
        tool = EnumerateFilesTool(context)

        result = await tool.execute(include_samples=True, samples_per_type=2)

        assert result["success"] is True
        assert "sample_uris" in result["by_extension"][".pdf"]
        assert len(result["by_extension"][".pdf"]["sample_uris"]) == 2

    @pytest.mark.asyncio
    async def test_enumerate_excludes_samples_when_disabled(self, mock_connector: AsyncMock):
        """Test that samples are excluded when disabled."""
        context = {"connector": mock_connector, "source_id": 1}
        tool = EnumerateFilesTool(context)

        result = await tool.execute(include_samples=False)

        assert result["success"] is True
        assert "sample_uris" not in result["by_extension"][".pdf"]

    @pytest.mark.asyncio
    async def test_enumerate_calculates_size_distribution(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test that size distribution is calculated correctly."""
        context = {"connector": mock_connector, "source_id": 1}
        tool = EnumerateFilesTool(context)

        result = await tool.execute()

        assert result["success"] is True
        assert "size_distribution" in result
        # Based on sample sizes: 100KB=100000, 200KB=200000, 5KB=5000, 2KB=2000
        # small (1KB-100KB): 5KB, 2KB, 100KB = 3 files
        # medium (100KB-1MB): 200KB = 1 file
        assert result["size_distribution"]["tiny"] == 0
        assert result["size_distribution"]["small"] == 3  # 5KB, 2KB, and 100KB (all < 100KB)
        assert result["size_distribution"]["medium"] == 1  # Only 200KB

    @pytest.mark.asyncio
    async def test_enumerate_stores_refs_in_context(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test that enumerated refs are stored in context for sampling."""
        context: dict[str, Any] = {"connector": mock_connector, "source_id": 1}
        tool = EnumerateFilesTool(context)

        await tool.execute()

        assert "_enumerated_files" in context
        assert len(context["_enumerated_files"]) == 4

    @pytest.mark.asyncio
    async def test_enumerate_fails_without_connector(self):
        """Test that enumeration fails without a connector."""
        context: dict[str, Any] = {}
        tool = EnumerateFilesTool(context)

        result = await tool.execute()

        assert result["success"] is False
        assert "error" in result


# SampleFilesTool tests
class TestSampleFilesTool:
    """Tests for the SampleFilesTool."""

    @pytest.mark.asyncio
    async def test_sample_returns_files(self, sample_file_refs: list[FileReference]):
        """Test that sampling returns files."""
        context: dict[str, Any] = {"_enumerated_files": sample_file_refs}
        tool = SampleFilesTool(context)

        result = await tool.execute(count=3)

        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["files"]) == 3

    @pytest.mark.asyncio
    async def test_sample_filters_by_extension(self, sample_file_refs: list[FileReference]):
        """Test that sampling filters by extension."""
        context: dict[str, Any] = {"_enumerated_files": sample_file_refs}
        tool = SampleFilesTool(context)

        result = await tool.execute(extension=".pdf")

        assert result["success"] is True
        assert result["total_matching"] == 2
        for f in result["files"]:
            assert f["extension"] == ".pdf"

    @pytest.mark.asyncio
    async def test_sample_filters_by_size_range(self, sample_file_refs: list[FileReference]):
        """Test that sampling filters by size range."""
        context: dict[str, Any] = {"_enumerated_files": sample_file_refs}
        tool = SampleFilesTool(context)

        result = await tool.execute(min_size_bytes=50000, max_size_bytes=150000)

        assert result["success"] is True
        assert result["total_matching"] == 1  # Only paper1.pdf (100000)
        assert result["files"][0]["filename"] == "paper1.pdf"

    @pytest.mark.asyncio
    async def test_sample_fails_without_enumeration(self):
        """Test that sampling fails without prior enumeration."""
        context: dict[str, Any] = {}
        tool = SampleFilesTool(context)

        result = await tool.execute()

        assert result["success"] is False
        assert "enumerate_files" in result["error"]


# TryParserTool tests
class TestTryParserTool:
    """Tests for the TryParserTool."""

    @pytest.mark.asyncio
    async def test_try_parser_basic(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test basic parser trial with fallback parsing."""
        mock_connector.load_content = AsyncMock(
            return_value=b"# Heading\n\nThis is some markdown content.\n\nMore text here."
        )
        context: dict[str, Any] = {
            "_enumerated_files": sample_file_refs,
            "connector": mock_connector,
        }
        tool = TryParserTool(context)

        result = await tool.execute(
            file_uri="file:///docs/notes.md",
            parser_id="text",
        )

        # Parser plugin doesn't have parse() method yet, so it falls back to basic text parsing
        # The fallback is triggered by the AttributeError caught in the try block
        assert result["success"] is True
        assert "stats" in result
        assert result["stats"]["text_length"] > 0
        # Fallback sets a flag in parse_metadata
        assert result["parse_metadata"].get("fallback") is True

    @pytest.mark.asyncio
    async def test_try_parser_file_not_found(self, mock_connector: AsyncMock):
        """Test parser trial with missing file."""
        context: dict[str, Any] = {
            "_enumerated_files": [],
            "connector": mock_connector,
        }
        tool = TryParserTool(context)

        result = await tool.execute(
            file_uri="file:///nonexistent.txt",
            parser_id="text",
        )

        assert result["success"] is False
        assert "not found" in result["error"]


# DetectLanguageTool tests
class TestDetectLanguageTool:
    """Tests for the DetectLanguageTool."""

    @pytest.mark.asyncio
    async def test_detect_language_from_text(self):
        """Test language detection from provided text."""
        context: dict[str, Any] = {}
        tool = DetectLanguageTool(context)

        result = await tool.execute(
            text="The quick brown fox jumps over the lazy dog. This is a test sentence."
        )

        assert result["success"] is True
        assert result["primary_language"] == "en"
        assert result["source"] == "provided_text"

    @pytest.mark.asyncio
    async def test_detect_language_from_file(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test language detection from file content."""
        mock_connector.load_content = AsyncMock(
            return_value=b"This is English text. The quick brown fox."
        )
        context: dict[str, Any] = {
            "_enumerated_files": sample_file_refs,
            "connector": mock_connector,
        }
        tool = DetectLanguageTool(context)

        result = await tool.execute(file_uri="file:///docs/notes.md")

        assert result["success"] is True
        assert result["primary_language"] == "en"

    @pytest.mark.asyncio
    async def test_detect_language_fails_without_input(self):
        """Test that detection fails without text or file."""
        context: dict[str, Any] = {}
        tool = DetectLanguageTool(context)

        result = await tool.execute()

        assert result["success"] is False
        assert "Either file_uri or text" in result["error"]


# GetFileContentPreviewTool tests
class TestGetFileContentPreviewTool:
    """Tests for the GetFileContentPreviewTool."""

    @pytest.mark.asyncio
    async def test_preview_text_file(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test previewing a text file."""
        test_content = b"This is the content of the file for preview."
        mock_connector.load_content = AsyncMock(return_value=test_content)
        context: dict[str, Any] = {
            "_enumerated_files": sample_file_refs,
            "connector": mock_connector,
        }
        tool = GetFileContentPreviewTool(context)

        result = await tool.execute(file_uri="file:///docs/notes.md")

        assert result["success"] is True
        assert result["content_type"] == "text"
        assert result["preview"] == test_content.decode("utf-8")
        assert result["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_preview_respects_max_bytes(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test that preview respects max_bytes limit."""
        long_content = b"x" * 5000
        mock_connector.load_content = AsyncMock(return_value=long_content)
        context: dict[str, Any] = {
            "_enumerated_files": sample_file_refs,
            "connector": mock_connector,
        }
        tool = GetFileContentPreviewTool(context)

        result = await tool.execute(file_uri="file:///docs/notes.md", max_bytes=100)

        assert result["success"] is True
        assert len(result["preview"]) == 100

    @pytest.mark.asyncio
    async def test_preview_file_not_found(self, mock_connector: AsyncMock):
        """Test preview with missing file."""
        context: dict[str, Any] = {
            "_enumerated_files": [],
            "connector": mock_connector,
        }
        tool = GetFileContentPreviewTool(context)

        result = await tool.execute(file_uri="file:///nonexistent.txt")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_preview_binary_file(
        self, mock_connector: AsyncMock, sample_file_refs: list[FileReference]
    ):
        """Test previewing a binary file."""
        # Binary content that won't decode as UTF-8
        binary_content = bytes([0x00, 0xFF, 0x89, 0x50, 0x4E, 0x47] * 100)
        mock_connector.load_content = AsyncMock(return_value=binary_content)
        context: dict[str, Any] = {
            "_enumerated_files": sample_file_refs,
            "connector": mock_connector,
        }
        tool = GetFileContentPreviewTool(context)

        result = await tool.execute(file_uri="file:///docs/paper1.pdf", as_text=False)

        assert result["success"] is True
        assert result["content_type"] == "binary"
        assert result["preview"] is None
        assert "preview_hex" in result


# Tool schema tests
class TestToolSchemas:
    """Tests for tool schema generation."""

    def test_enumerate_tool_schema(self):
        """Test EnumerateFilesTool schema."""
        tool = EnumerateFilesTool({})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "enumerate_files"
        assert "include_samples" in schema["function"]["parameters"]["properties"]

    def test_sample_tool_schema(self):
        """Test SampleFilesTool schema."""
        tool = SampleFilesTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "sample_files"
        assert "extension" in schema["function"]["parameters"]["properties"]
        assert "min_size_bytes" in schema["function"]["parameters"]["properties"]

    def test_try_parser_tool_schema(self):
        """Test TryParserTool schema."""
        tool = TryParserTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "try_parser"
        assert "file_uri" in schema["function"]["parameters"]["required"]
        assert "parser_id" in schema["function"]["parameters"]["required"]

    def test_detect_language_tool_schema(self):
        """Test DetectLanguageTool schema."""
        tool = DetectLanguageTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "detect_language"
        assert "file_uri" in schema["function"]["parameters"]["properties"]
        assert "text" in schema["function"]["parameters"]["properties"]

    def test_preview_tool_schema(self):
        """Test GetFileContentPreviewTool schema."""
        tool = GetFileContentPreviewTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "get_file_content_preview"
        assert "file_uri" in schema["function"]["parameters"]["required"]
        assert "max_bytes" in schema["function"]["parameters"]["properties"]
