"""Unit tests for the SourceAnalyzer sub-agent.

These tests verify the SourceAnalyzer's ability to analyze data sources
and produce structured recommendations.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.pipeline.types import FileReference
from webui.services.agent.subagents.base import (
    Message,
    SubAgentResult,
    Uncertainty,
)
from webui.services.agent.subagents.source_analyzer import (
    ContentCharacteristics,
    FileTypeStats,
    ParserRecommendation,
    SourceAnalysis,
    SourceAnalyzer,
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
    ]


@pytest.fixture
def mock_connector(sample_file_refs: list[FileReference]) -> AsyncMock:
    """Create a mock connector."""
    connector = AsyncMock()
    connector.authenticate = AsyncMock(return_value=True)

    async def mock_enumerate(source_id: int | None = None):
        for ref in sample_file_refs:
            yield ref

    connector.enumerate = mock_enumerate
    connector.load_content = AsyncMock(return_value=b"Sample content")
    return connector


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.generate = AsyncMock()
    provider.__aenter__ = AsyncMock(return_value=provider)
    provider.__aexit__ = AsyncMock(return_value=None)
    return provider


@pytest.fixture
def analyzer_context(mock_connector: AsyncMock) -> dict[str, Any]:
    """Create context for SourceAnalyzer."""
    return {
        "source_id": 1,
        "connector": mock_connector,
        "user_intent": "Search my research papers",
    }


# SourceAnalysis dataclass tests
class TestSourceAnalysis:
    """Tests for the SourceAnalysis dataclass."""

    def test_to_dict_basic(self):
        """Test basic serialization."""
        analysis = SourceAnalysis(
            total_files=10,
            total_size_bytes=1000000,
            by_extension={
                ".pdf": FileTypeStats(count=8, total_size_bytes=800000),
                ".md": FileTypeStats(count=2, total_size_bytes=200000),
            },
            content_characteristics=ContentCharacteristics(
                languages=["en"],
                document_types=["academic"],
                quality_issues=[],
            ),
            parser_recommendations=[
                ParserRecommendation(
                    extension=".pdf",
                    parser_id="unstructured",
                    confidence=0.9,
                    notes="Good for complex layouts",
                ),
            ],
            uncertainties=[
                Uncertainty(
                    severity="notable",
                    message="Some files may be scanned",
                ),
            ],
            summary="10 files analyzed",
        )

        result = analysis.to_dict()

        assert result["total_files"] == 10
        assert result["total_size_bytes"] == 1000000
        assert ".pdf" in result["by_extension"]
        assert result["by_extension"][".pdf"]["count"] == 8
        assert result["content_characteristics"]["languages"] == ["en"]
        assert len(result["parser_recommendations"]) == 1
        assert result["parser_recommendations"][0]["parser_id"] == "unstructured"
        assert len(result["uncertainties"]) == 1
        assert result["uncertainties"][0]["severity"] == "notable"


# SourceAnalyzer tests
class TestSourceAnalyzer:
    """Tests for the SourceAnalyzer sub-agent."""

    def test_class_attributes(self):
        """Test class attributes are set correctly."""
        assert SourceAnalyzer.AGENT_ID == "source_analyzer"
        assert SourceAnalyzer.MAX_TURNS == 30
        assert SourceAnalyzer.TIMEOUT_SECONDS == 300
        assert len(SourceAnalyzer.TOOLS) == 5

    def test_initialization(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test agent initialization."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)

        assert agent.llm == mock_llm_provider
        assert agent.context == analyzer_context
        assert len(agent.tools) == 5
        assert "enumerate_files" in agent.tools
        assert "sample_files" in agent.tools

    def test_build_initial_message(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test initial message building."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        message = agent._build_initial_message()

        assert message.role == "user"
        assert "1" in message.content  # source_id
        assert "Search my research papers" in message.content

    def test_build_initial_message_without_intent(
        self, mock_llm_provider: AsyncMock, mock_connector: AsyncMock
    ):
        """Test initial message without user intent."""
        context = {"source_id": 1, "connector": mock_connector}
        agent = SourceAnalyzer(mock_llm_provider, context)
        message = agent._build_initial_message()

        assert "Not specified" in message.content

    def test_parse_analysis_json_from_code_block(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test parsing JSON from markdown code block."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        content = """Here's my analysis:

```json
{
    "total_files": 50,
    "summary": "Found 50 files"
}
```

That's my analysis."""

        result = agent._parse_analysis_json(content)

        assert result is not None
        assert result["total_files"] == 50
        assert result["summary"] == "Found 50 files"

    def test_parse_analysis_json_raw(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test parsing raw JSON."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        content = '{"total_files": 25, "summary": "25 files"}'

        result = agent._parse_analysis_json(content)

        assert result is not None
        assert result["total_files"] == 25

    def test_parse_analysis_json_embedded(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test parsing embedded JSON object."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        content = 'Here is the analysis: {"total_files": 100} Done.'

        result = agent._parse_analysis_json(content)

        assert result is not None
        assert result["total_files"] == 100

    def test_parse_analysis_json_invalid(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test parsing invalid content."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        content = "This is not JSON at all"

        result = agent._parse_analysis_json(content)

        assert result is None

    def test_extract_result_success(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test successful result extraction."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        response = Message(
            role="assistant",
            content=json.dumps(
                {
                    "total_files": 100,
                    "total_size_bytes": 1000000,
                    "by_extension": {".pdf": {"count": 80}},
                    "content_characteristics": {
                        "languages": ["en"],
                        "document_types": ["academic"],
                        "quality_issues": [],
                    },
                    "parser_recommendations": [],
                    "uncertainties": [
                        {"severity": "notable", "message": "Some uncertainty"}
                    ],
                    "summary": "Analysis complete",
                }
            ),
        )

        result = agent._extract_result(response)

        assert result.success is True
        assert result.data["total_files"] == 100
        assert result.summary == "Analysis complete"
        assert len(result.uncertainties) == 1
        assert result.uncertainties[0].severity == "notable"

    def test_extract_result_invalid_json(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test result extraction with invalid JSON."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        response = Message(
            role="assistant",
            content="This is not valid JSON",
        )

        result = agent._extract_result(response)

        assert result.success is False

    def test_get_partial_result_with_files(
        self,
        mock_llm_provider: AsyncMock,
        analyzer_context: dict[str, Any],
        sample_file_refs: list[FileReference],
    ):
        """Test partial result with enumerated files."""
        analyzer_context["_enumerated_files"] = sample_file_refs
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)

        partial = agent._get_partial_result()

        assert partial["partial"] is True
        assert partial["total_files"] == 2
        assert ".pdf" in partial["by_extension"]
        assert ".md" in partial["by_extension"]

    def test_get_partial_result_empty(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test partial result without enumerated files."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)

        partial = agent._get_partial_result()

        assert partial["partial"] is True
        assert partial["total_files"] == 0

    def test_get_tool_schemas(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test tool schema generation."""
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        schemas = agent.get_tool_schemas()

        assert len(schemas) == 5
        tool_names = {s["function"]["name"] for s in schemas}
        assert "enumerate_files" in tool_names
        assert "sample_files" in tool_names
        assert "try_parser" in tool_names
        assert "detect_language" in tool_names
        assert "get_file_content_preview" in tool_names


# Integration-like tests (still mocked, but testing the flow)
class TestSourceAnalyzerFlow:
    """Tests for the SourceAnalyzer execution flow."""

    @pytest.mark.asyncio
    async def test_run_handles_timeout(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test that run handles timeout gracefully."""
        # Make generate hang forever
        async def hang_forever(*args, **kwargs):
            import asyncio

            await asyncio.sleep(1000)

        mock_llm_provider.generate = hang_forever

        # Create agent with very short timeout for testing
        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)
        agent.TIMEOUT_SECONDS = 0.1  # Very short timeout

        result = await agent.run()

        assert result.success is False
        assert "Timed out" in result.summary

    @pytest.mark.asyncio
    async def test_run_handles_exceptions(
        self, mock_llm_provider: AsyncMock, analyzer_context: dict[str, Any]
    ):
        """Test that run handles exceptions gracefully."""
        mock_llm_provider.generate = AsyncMock(side_effect=RuntimeError("LLM error"))

        agent = SourceAnalyzer(mock_llm_provider, analyzer_context)

        result = await agent.run()

        assert result.success is False
        assert "error" in result.summary.lower()
