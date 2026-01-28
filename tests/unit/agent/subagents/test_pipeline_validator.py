"""Unit tests for the PipelineValidator sub-agent.

These tests verify the PipelineValidator's ability to validate pipeline
configurations and produce structured validation reports.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from shared.pipeline.types import FileReference
from webui.services.agent.subagents.base import (
    Message,
)
from webui.services.agent.subagents.pipeline_validator import (
    FailureCategory,
    PipelineFix,
    PipelineValidator,
    ValidationReport,
)


# Fixtures
@pytest.fixture()
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
            metadata={"source": {"local_path": "/docs/paper1.pdf"}},
        ),
        FileReference(
            uri="file:///docs/notes.md",
            source_type="local_directory",
            content_type="file",
            filename="notes.md",
            extension=".md",
            mime_type="text/markdown",
            size_bytes=5000,
            metadata={"source": {"local_path": "/docs/notes.md"}},
        ),
    ]


@pytest.fixture()
def sample_pipeline() -> dict[str, Any]:
    """Create a sample pipeline configuration."""
    return {
        "id": "test-pipeline",
        "version": "1",
        "nodes": [
            {"id": "parser", "type": "parser", "plugin_id": "text", "config": {}},
            {"id": "chunker", "type": "chunker", "plugin_id": "character", "config": {}},
            {"id": "embedder", "type": "embedder", "plugin_id": "mock", "config": {}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "parser"},
            {"from_node": "parser", "to_node": "chunker"},
            {"from_node": "chunker", "to_node": "embedder"},
        ],
    }


@pytest.fixture()
def mock_connector() -> AsyncMock:
    """Create a mock connector."""
    connector = AsyncMock()
    connector.authenticate = AsyncMock(return_value=True)
    connector.load_content = AsyncMock(return_value=b"Sample content")
    return connector


@pytest.fixture()
def mock_llm_provider() -> AsyncMock:
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.generate = AsyncMock()
    provider.__aenter__ = AsyncMock(return_value=provider)
    provider.__aexit__ = AsyncMock(return_value=None)
    return provider


@pytest.fixture()
def validator_context(
    mock_connector: AsyncMock,
    sample_pipeline: dict[str, Any],
    sample_file_refs: list[FileReference],
) -> dict[str, Any]:
    """Create context for PipelineValidator."""
    return {
        "pipeline": sample_pipeline,
        "sample_files": sample_file_refs,
        "connector": mock_connector,
        "session": AsyncMock(),
    }


# ValidationReport dataclass tests
class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_to_dict_basic(self):
        """Test basic serialization."""
        report = ValidationReport(
            success_rate=0.95,
            files_tested=100,
            files_passed=95,
            files_failed=5,
            assessment="ready",
            failure_categories=[
                FailureCategory(
                    category="encoding_error",
                    count=5,
                    example_files=["file1.pdf", "file2.pdf"],
                    is_fixable=False,
                ),
            ],
            suggested_fixes=[
                PipelineFix(
                    issue="5 files have encoding issues",
                    fix_type="accept",
                    details={},
                    affected_files=5,
                    confidence=0.9,
                ),
            ],
            chunk_quality={
                "avg_chunk_size": 450,
                "size_variance": "normal",
            },
            summary="Pipeline ready for production",
        )

        result = report.to_dict()

        assert result["success_rate"] == 0.95
        assert result["files_tested"] == 100
        assert result["files_passed"] == 95
        assert result["files_failed"] == 5
        assert result["assessment"] == "ready"
        assert len(result["failure_categories"]) == 1
        assert result["failure_categories"][0]["category"] == "encoding_error"
        assert len(result["suggested_fixes"]) == 1
        assert result["suggested_fixes"][0]["fix_type"] == "accept"
        assert result["chunk_quality"]["avg_chunk_size"] == 450
        assert result["summary"] == "Pipeline ready for production"

    def test_to_dict_empty_lists(self):
        """Test serialization with empty lists."""
        report = ValidationReport(
            success_rate=1.0,
            files_tested=50,
            files_passed=50,
            files_failed=0,
            assessment="ready",
            failure_categories=[],
            suggested_fixes=[],
            summary="Perfect run",
        )

        result = report.to_dict()

        assert result["failure_categories"] == []
        assert result["suggested_fixes"] == []


# FailureCategory tests
class TestFailureCategory:
    """Tests for the FailureCategory dataclass."""

    def test_creation(self):
        """Test creating a failure category."""
        category = FailureCategory(
            category="parser_error",
            count=10,
            example_files=["a.pdf", "b.pdf"],
            is_fixable=True,
            suggested_fix="Try unstructured parser",
        )

        assert category.category == "parser_error"
        assert category.count == 10
        assert len(category.example_files) == 2
        assert category.is_fixable is True
        assert category.suggested_fix == "Try unstructured parser"


# PipelineFix tests
class TestPipelineFix:
    """Tests for the PipelineFix dataclass."""

    def test_creation(self):
        """Test creating a pipeline fix."""
        fix = PipelineFix(
            issue="PDF parsing fails on scanned documents",
            fix_type="parser_change",
            details={"new_parser": "unstructured", "old_parser": "text"},
            affected_files=15,
            confidence=0.85,
        )

        assert fix.issue == "PDF parsing fails on scanned documents"
        assert fix.fix_type == "parser_change"
        assert fix.details["new_parser"] == "unstructured"
        assert fix.affected_files == 15
        assert fix.confidence == 0.85


# PipelineValidator tests
class TestPipelineValidator:
    """Tests for the PipelineValidator sub-agent."""

    def test_class_attributes(self):
        """Test class attributes are set correctly."""
        assert PipelineValidator.AGENT_ID == "pipeline_validator"
        assert PipelineValidator.MAX_TURNS == 25
        assert PipelineValidator.TIMEOUT_SECONDS == 300
        assert len(PipelineValidator.TOOLS) == 5

    def test_initialization(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test agent initialization."""
        agent = PipelineValidator(mock_llm_provider, validator_context)

        assert agent.llm == mock_llm_provider
        assert agent.context == validator_context
        assert len(agent.tools) == 5
        assert "run_dry_run" in agent.tools
        assert "get_failure_details" in agent.tools
        assert "try_alternative_config" in agent.tools
        assert "compare_parser_output" in agent.tools
        assert "inspect_chunks" in agent.tools

    def test_build_initial_message(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
        sample_pipeline: dict[str, Any],
    ):
        """Test initial message building."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        message = agent._build_initial_message()

        assert message.role == "user"
        assert sample_pipeline["id"] in message.content
        assert "3 nodes" in message.content  # Pipeline has 3 nodes
        assert "2 sample files" in message.content

    def test_build_initial_message_shows_extensions(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """Test initial message includes file extension summary."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        message = agent._build_initial_message()

        # Should include extension summary
        assert ".pdf" in message.content or "pdf" in message.content.lower()

    def test_parse_report_json_from_code_block(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test parsing JSON from markdown code block."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        content = """Here's my validation report:

```json
{
    "success_rate": 0.95,
    "files_tested": 50,
    "files_passed": 48,
    "files_failed": 2,
    "assessment": "ready",
    "summary": "Pipeline ready"
}
```

That's the report."""

        result = agent._parse_report_json(content)

        assert result is not None
        assert result["success_rate"] == 0.95
        assert result["assessment"] == "ready"

    def test_parse_report_json_raw(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test parsing raw JSON."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        content = '{"success_rate": 0.9, "assessment": "needs_review"}'

        result = agent._parse_report_json(content)

        assert result is not None
        assert result["success_rate"] == 0.9

    def test_parse_report_json_embedded(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test parsing embedded JSON object."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        content = 'The report is: {"success_rate": 1.0, "assessment": "ready"} End.'

        result = agent._parse_report_json(content)

        assert result is not None
        assert result["success_rate"] == 1.0

    def test_parse_report_json_invalid(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test parsing invalid content."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        content = "This is not JSON at all"

        result = agent._parse_report_json(content)

        assert result is None

    def test_extract_result_success_ready(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test successful result extraction with ready assessment."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content=json.dumps(
                {
                    "success_rate": 0.98,
                    "files_tested": 100,
                    "files_passed": 98,
                    "files_failed": 2,
                    "assessment": "ready",
                    "failure_categories": [],
                    "suggested_fixes": [],
                    "chunk_quality": {"avg_chunk_size": 400},
                    "summary": "Pipeline is production ready",
                }
            ),
        )

        result = agent._extract_result(response)

        assert result.success is True
        assert result.data["success_rate"] == 0.98
        assert result.data["assessment"] == "ready"
        assert result.summary == "Pipeline is production ready"
        # No uncertainties for >95% success
        assert len(result.uncertainties) == 0

    def test_extract_result_needs_review(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test result extraction with needs_review assessment."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content=json.dumps(
                {
                    "success_rate": 0.92,
                    "files_tested": 100,
                    "files_passed": 92,
                    "files_failed": 8,
                    "assessment": "needs_review",
                    "failure_categories": [{"category": "encoding_error", "count": 8, "is_fixable": False}],
                    "suggested_fixes": [],
                    "summary": "Some issues to review",
                }
            ),
        )

        result = agent._extract_result(response)

        assert result.success is True
        # Should have notable uncertainty for 90-95%
        notable_uncertainties = [u for u in result.uncertainties if u.severity == "notable"]
        assert len(notable_uncertainties) >= 1

    def test_extract_result_blocking_issues(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test result extraction with blocking_issues assessment."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content=json.dumps(
                {
                    "success_rate": 0.75,
                    "files_tested": 100,
                    "files_passed": 75,
                    "files_failed": 25,
                    "assessment": "blocking_issues",
                    "failure_categories": [{"category": "parser_error", "count": 25, "is_fixable": True}],
                    "suggested_fixes": [],
                    "summary": "Major issues",
                }
            ),
        )

        result = agent._extract_result(response)

        assert result.success is True
        # Should have blocking uncertainty for <90%
        blocking_uncertainties = [u for u in result.uncertainties if u.severity == "blocking"]
        assert len(blocking_uncertainties) >= 1

    def test_extract_result_adds_uncertainties_for_non_fixable_failures(
        self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]
    ):
        """Test that non-fixable failures create info uncertainties."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content=json.dumps(
                {
                    "success_rate": 0.96,
                    "files_tested": 100,
                    "files_passed": 96,
                    "files_failed": 4,
                    "assessment": "ready",
                    "failure_categories": [
                        {
                            "category": "corrupted_file",
                            "count": 4,
                            "is_fixable": False,
                            "example_files": ["file1.pdf", "file2.pdf"],
                        }
                    ],
                    "suggested_fixes": [],
                    "summary": "Ready with some corrupted files",
                }
            ),
        )

        result = agent._extract_result(response)

        # Should have info uncertainty for non-fixable failures
        info_uncertainties = [u for u in result.uncertainties if u.severity == "info"]
        assert len(info_uncertainties) >= 1
        assert "corrupted_file" in info_uncertainties[0].message

    def test_extract_result_invalid_json(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test result extraction with invalid JSON."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content="This is not valid JSON",
        )

        result = agent._extract_result(response)

        assert result.success is False
        assert "Could not extract" in result.summary

    def test_get_partial_result_with_dry_run(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test partial result with dry-run data."""
        # Mock a dry-run result
        from unittest.mock import MagicMock

        mock_dry_run = MagicMock()
        mock_dry_run.files_succeeded = 45
        mock_dry_run.files_failed = 5
        validator_context["_dry_run_result"] = mock_dry_run

        agent = PipelineValidator(mock_llm_provider, validator_context)

        partial = agent._get_partial_result()

        assert partial["partial"] is True
        assert partial["success_rate"] == 0.9  # 45/50
        assert partial["files_tested"] == 50
        assert partial["assessment"] == "needs_review"  # 90% exactly

    def test_get_partial_result_empty(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test partial result without dry-run data."""
        agent = PipelineValidator(mock_llm_provider, validator_context)

        partial = agent._get_partial_result()

        assert partial["partial"] is True
        assert partial["success_rate"] == 0
        assert partial["files_tested"] == 0

    def test_get_tool_schemas(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test tool schema generation."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        schemas = agent.get_tool_schemas()

        assert len(schemas) == 5
        tool_names = {s["function"]["name"] for s in schemas}
        assert "run_dry_run" in tool_names
        assert "get_failure_details" in tool_names
        assert "try_alternative_config" in tool_names
        assert "compare_parser_output" in tool_names
        assert "inspect_chunks" in tool_names


# Integration-like tests (still mocked, but testing the flow)
class TestPipelineValidatorFlow:
    """Tests for the PipelineValidator execution flow."""

    @pytest.mark.asyncio()
    async def test_run_handles_timeout(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test that run handles timeout gracefully."""

        # Make generate hang forever
        async def hang_forever(*_args, **_kwargs):
            import asyncio

            await asyncio.sleep(1000)

        mock_llm_provider.generate = hang_forever

        # Create agent with very short timeout for testing
        agent = PipelineValidator(mock_llm_provider, validator_context)
        agent.TIMEOUT_SECONDS = 0.1  # Very short timeout

        result = await agent.run()

        assert result.success is False
        assert "Timed out" in result.summary

    @pytest.mark.asyncio()
    async def test_run_handles_exceptions(self, mock_llm_provider: AsyncMock, validator_context: dict[str, Any]):
        """Test that run handles exceptions gracefully."""
        mock_llm_provider.generate = AsyncMock(side_effect=RuntimeError("LLM error"))

        agent = PipelineValidator(mock_llm_provider, validator_context)

        result = await agent.run()

        assert result.success is False
        assert "error" in result.summary.lower()
