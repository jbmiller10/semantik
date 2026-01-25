"""Integration tests for the PipelineValidator sub-agent.

These tests verify the PipelineValidator tools work correctly with
real database sessions but mocked connectors and LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from shared.pipeline.types import FileReference
from shared.utils.encryption import SecretEncryption, generate_fernet_key
from webui.services.agent.subagents.base import Message
from webui.services.agent.subagents.pipeline_validator import (
    FailureCategory,
    PipelineFix,
    PipelineValidator,
    ValidationReport,
)
from webui.services.agent.tools.subagent_tools.validation import (
    GetFailureDetailsTool,
    InspectChunksTool,
    RunDryRunTool,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture(autouse=True)
def _initialize_encryption():
    """Initialize encryption for all tests."""
    test_key = generate_fernet_key()
    SecretEncryption.initialize(test_key)
    yield
    SecretEncryption.reset()


@pytest.fixture()
def sample_file_refs() -> list[FileReference]:
    """Create sample file references for testing."""
    return [
        FileReference(
            uri="file:///test/doc1.txt",
            source_type="local_directory",
            content_type="file",
            filename="doc1.txt",
            extension=".txt",
            mime_type="text/plain",
            size_bytes=1000,
            source_metadata={"local_path": "/test/doc1.txt"},
        ),
        FileReference(
            uri="file:///test/doc2.md",
            source_type="local_directory",
            content_type="file",
            filename="doc2.md",
            extension=".md",
            mime_type="text/markdown",
            size_bytes=2000,
            source_metadata={"local_path": "/test/doc2.md"},
        ),
        FileReference(
            uri="file:///test/doc3.pdf",
            source_type="local_directory",
            content_type="file",
            filename="doc3.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=50000,
            source_metadata={"local_path": "/test/doc3.pdf"},
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
            {
                "id": "chunker",
                "type": "chunker",
                "plugin_id": "character",
                "config": {"max_tokens": 500},
            },
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
    """Create a mock connector that returns sample content."""
    connector = AsyncMock()
    connector.authenticate = AsyncMock(return_value=True)

    def return_content(file_ref: FileReference) -> bytes:
        """Return appropriate content based on file extension."""
        if file_ref.extension == ".txt":
            return b"This is a plain text document with some sample content."
        if file_ref.extension == ".md":
            return b"# Markdown Document\n\nThis is **markdown** content."
        if file_ref.extension == ".pdf":
            # Return some bytes that would fail with text parser
            return b"%PDF-1.4 binary content"
        return b"Generic content"

    connector.load_content = AsyncMock(side_effect=return_content)
    return connector


@pytest.fixture()
def mock_llm_provider() -> AsyncMock:
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.__aenter__ = AsyncMock(return_value=provider)
    provider.__aexit__ = AsyncMock(return_value=None)
    return provider


class TestValidationReportSerialization:
    """Tests for ValidationReport dataclass serialization."""

    def test_to_dict_includes_all_fields(self):
        """ValidationReport.to_dict() serializes all fields correctly."""
        report = ValidationReport(
            success_rate=0.90,
            files_tested=50,
            files_passed=45,
            files_failed=5,
            assessment="needs_review",
            failure_categories=[
                FailureCategory(
                    category="parser_error",
                    count=3,
                    example_files=["a.pdf", "b.pdf"],
                    is_fixable=True,
                    suggested_fix="Try unstructured parser",
                ),
                FailureCategory(
                    category="encoding_error",
                    count=2,
                    example_files=["c.txt"],
                    is_fixable=False,
                ),
            ],
            suggested_fixes=[
                PipelineFix(
                    issue="Parser incompatible with PDF",
                    fix_type="parser_change",
                    details={"new_parser": "unstructured"},
                    affected_files=3,
                    confidence=0.85,
                ),
            ],
            chunk_quality={"avg_chunk_size": 400, "variance": "normal"},
            summary="Pipeline needs review due to 5 failures",
        )

        result = report.to_dict()

        assert result["success_rate"] == 0.90
        assert result["files_tested"] == 50
        assert result["files_passed"] == 45
        assert result["files_failed"] == 5
        assert result["assessment"] == "needs_review"
        assert len(result["failure_categories"]) == 2
        assert result["failure_categories"][0]["category"] == "parser_error"
        assert result["failure_categories"][0]["is_fixable"] is True
        assert result["failure_categories"][1]["category"] == "encoding_error"
        assert result["failure_categories"][1]["is_fixable"] is False
        assert len(result["suggested_fixes"]) == 1
        assert result["suggested_fixes"][0]["fix_type"] == "parser_change"
        assert result["chunk_quality"]["avg_chunk_size"] == 400
        assert result["summary"] == "Pipeline needs review due to 5 failures"


class TestPipelineValidatorResultExtraction:
    """Tests for PipelineValidator result extraction."""

    @pytest.fixture()
    def validator_context(
        self,
        mock_connector: AsyncMock,
        sample_pipeline: dict[str, Any],
        sample_file_refs: list[FileReference],
        db_session: AsyncSession,
    ) -> dict[str, Any]:
        """Create context for PipelineValidator with real DB session."""
        return {
            "pipeline": sample_pipeline,
            "sample_files": sample_file_refs,
            "connector": mock_connector,
            "session": db_session,
        }

    def test_extract_result_creates_blocking_uncertainty(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """Blocking issues create blocking uncertainty."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content="""{
                "success_rate": 0.70,
                "files_tested": 100,
                "files_passed": 70,
                "files_failed": 30,
                "assessment": "blocking_issues",
                "failure_categories": [],
                "suggested_fixes": [],
                "summary": "Too many failures"
            }""",
        )

        result = agent._extract_result(response)

        assert result.success is True
        blocking = [u for u in result.uncertainties if u.severity == "blocking"]
        assert len(blocking) == 1
        assert "70%" in blocking[0].message

    def test_extract_result_creates_notable_uncertainty(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """Needs review creates notable uncertainty."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content="""{
                "success_rate": 0.92,
                "files_tested": 100,
                "files_passed": 92,
                "files_failed": 8,
                "assessment": "needs_review",
                "failure_categories": [
                    {"category": "encoding_error", "count": 8, "is_fixable": false}
                ],
                "suggested_fixes": [],
                "summary": "Review needed"
            }""",
        )

        result = agent._extract_result(response)

        assert result.success is True
        notable = [u for u in result.uncertainties if u.severity == "notable"]
        assert len(notable) >= 1

    def test_extract_result_creates_info_for_nonfixable(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """Non-fixable failures create info uncertainties."""
        agent = PipelineValidator(mock_llm_provider, validator_context)
        response = Message(
            role="assistant",
            content="""{
                "success_rate": 0.96,
                "files_tested": 100,
                "files_passed": 96,
                "files_failed": 4,
                "assessment": "ready",
                "failure_categories": [
                    {
                        "category": "corrupted_file",
                        "count": 4,
                        "is_fixable": false,
                        "example_files": ["bad.pdf"]
                    }
                ],
                "suggested_fixes": [],
                "summary": "Ready with some corrupted files"
            }""",
        )

        result = agent._extract_result(response)

        assert result.success is True
        info = [u for u in result.uncertainties if u.severity == "info"]
        assert len(info) >= 1
        assert any("corrupted_file" in u.message for u in info)


class TestRunDryRunToolIntegration:
    """Integration tests for RunDryRunTool with real database session."""

    @pytest_asyncio.fixture
    async def tool_context(
        self,
        db_session: AsyncSession,
        mock_connector: AsyncMock,
        sample_file_refs: list[FileReference],
        sample_pipeline: dict[str, Any],
    ) -> dict[str, Any]:
        """Create context for RunDryRunTool."""
        return {
            "session": db_session,
            "connector": mock_connector,
            "sample_files": sample_file_refs,
            "pipeline": sample_pipeline,
        }

    @pytest.mark.asyncio()
    async def test_returns_error_without_session(
        self,
        mock_connector: AsyncMock,
        sample_file_refs: list[FileReference],
        sample_pipeline: dict[str, Any],
    ):
        """RunDryRunTool returns error when no session provided."""
        context: dict[str, Any] = {
            "connector": mock_connector,
            "sample_files": sample_file_refs,
        }
        tool = RunDryRunTool(context)

        result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is False
        assert "session" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_returns_error_without_connector(
        self,
        db_session: AsyncSession,
        sample_file_refs: list[FileReference],
        sample_pipeline: dict[str, Any],
    ):
        """RunDryRunTool returns error when no connector provided."""
        context: dict[str, Any] = {
            "session": db_session,
            "sample_files": sample_file_refs,
        }
        tool = RunDryRunTool(context)

        result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is False
        assert "connector" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_returns_error_without_sample_files(
        self,
        db_session: AsyncSession,
        mock_connector: AsyncMock,
        sample_pipeline: dict[str, Any],
    ):
        """RunDryRunTool returns error when no sample files provided."""
        context: dict[str, Any] = {
            "session": db_session,
            "connector": mock_connector,
        }
        tool = RunDryRunTool(context)

        result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is False
        assert "sample files" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_returns_error_for_invalid_pipeline(
        self,
        tool_context: dict[str, Any],
    ):
        """RunDryRunTool returns error for invalid pipeline config."""
        tool = RunDryRunTool(tool_context)

        invalid_pipeline = {"id": "invalid", "nodes": [], "edges": []}
        result = await tool.execute(pipeline=invalid_pipeline)

        assert result["success"] is False
        # Either validation error or execution error
        assert "error" in result


class TestGetFailureDetailsTool:
    """Tests for GetFailureDetailsTool."""

    @pytest.fixture()
    def context_with_failures(self) -> dict[str, Any]:
        """Create context with simulated failures."""
        # Create a mock failure
        mock_failure = MagicMock()
        mock_failure.file_uri = "file:///test/failed.pdf"
        mock_failure.stage_id = "parser"
        mock_failure.stage_type = "parser"
        mock_failure.error_type = "ParseError"
        mock_failure.error_message = "Could not parse PDF"
        mock_failure.error_traceback = "Traceback..."

        return {
            "_dry_run_failures": {"file:///test/failed.pdf": mock_failure},
            "_dry_run_result": MagicMock(sample_outputs=[]),
        }

    @pytest.mark.asyncio()
    async def test_returns_failure_details(self, context_with_failures: dict[str, Any]):
        """GetFailureDetailsTool returns failure details."""
        tool = GetFailureDetailsTool(context_with_failures)

        result = await tool.execute(file_uri="file:///test/failed.pdf")

        assert result["success"] is True
        assert result["file_uri"] == "file:///test/failed.pdf"
        assert result["stage_type"] == "parser"
        assert result["error_type"] == "ParseError"
        assert result["error_message"] == "Could not parse PDF"

    @pytest.mark.asyncio()
    async def test_returns_error_for_unknown_file(self, context_with_failures: dict[str, Any]):
        """GetFailureDetailsTool returns error for unknown file."""
        tool = GetFailureDetailsTool(context_with_failures)

        result = await tool.execute(file_uri="file:///unknown.txt")

        assert result["success"] is False
        assert "not found" in result["error"].lower() or "No failure found" in result["error"]

    @pytest.mark.asyncio()
    async def test_returns_error_without_dry_run(self):
        """GetFailureDetailsTool returns error when no dry-run was performed."""
        tool = GetFailureDetailsTool({})

        result = await tool.execute(file_uri="file:///test.txt")

        assert result["success"] is False
        assert "dry-run" in result["error"].lower()


class TestInspectChunksTool:
    """Tests for InspectChunksTool."""

    @pytest.fixture()
    def context_with_chunks(self) -> dict[str, Any]:
        """Create context with simulated chunk output."""
        mock_sample = MagicMock()
        mock_sample.chunks = [
            {"content": "First chunk content" * 10, "metadata": {"token_count": 50}},
            {"content": "Second chunk content" * 10, "metadata": {"token_count": 55}},
            {"content": "Third chunk" * 5, "metadata": {"token_count": 25}},
        ]
        mock_sample.parse_metadata = {"title": "Test Document"}

        return {
            "_sample_outputs": {"file:///test/doc.txt": mock_sample},
        }

    @pytest.mark.asyncio()
    async def test_returns_chunk_statistics(self, context_with_chunks: dict[str, Any]):
        """InspectChunksTool returns chunk statistics."""
        tool = InspectChunksTool(context_with_chunks)

        result = await tool.execute(file_uri="file:///test/doc.txt")

        assert result["success"] is True
        assert result["chunk_count"] == 3
        assert "size_stats" in result
        assert "avg_chars" in result["size_stats"]

    @pytest.mark.asyncio()
    async def test_returns_previews(self, context_with_chunks: dict[str, Any]):
        """InspectChunksTool returns chunk previews."""
        tool = InspectChunksTool(context_with_chunks)

        result = await tool.execute(file_uri="file:///test/doc.txt", preview_count=2)

        assert result["success"] is True
        assert "previews" in result
        assert len(result["previews"]) == 2

    @pytest.mark.asyncio()
    async def test_returns_token_stats_when_available(self, context_with_chunks: dict[str, Any]):
        """InspectChunksTool includes token stats when available."""
        tool = InspectChunksTool(context_with_chunks)

        result = await tool.execute(file_uri="file:///test/doc.txt")

        assert result["success"] is True
        assert "token_stats" in result
        assert "avg_tokens" in result["token_stats"]

    @pytest.mark.asyncio()
    async def test_returns_error_without_sample_outputs(self):
        """InspectChunksTool returns error when no sample outputs available."""
        tool = InspectChunksTool({})

        result = await tool.execute(file_uri="file:///test.txt")

        assert result["success"] is False
        assert "dry_run" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_returns_error_for_unknown_file(self, context_with_chunks: dict[str, Any]):
        """InspectChunksTool returns error for unknown file."""
        tool = InspectChunksTool(context_with_chunks)

        result = await tool.execute(file_uri="file:///unknown.txt")

        assert result["success"] is False


class TestFailureCategorization:
    """Tests for failure categorization logic."""

    def test_failure_category_defaults(self):
        """FailureCategory has appropriate defaults."""
        category = FailureCategory(category="test", count=1)

        assert category.example_files == []
        assert category.is_fixable is False
        assert category.suggested_fix is None

    def test_pipeline_fix_parser_change(self):
        """PipelineFix supports parser_change fix type."""
        fix = PipelineFix(
            issue="Test issue",
            fix_type="parser_change",
            details={},
            affected_files=1,
            confidence=0.5,
        )
        assert fix.fix_type == "parser_change"

    def test_pipeline_fix_config_change(self):
        """PipelineFix supports config_change fix type."""
        fix = PipelineFix(
            issue="Test issue",
            fix_type="config_change",
            details={},
            affected_files=1,
            confidence=0.5,
        )
        assert fix.fix_type == "config_change"

    def test_pipeline_fix_filter_files(self):
        """PipelineFix supports filter_files fix type."""
        fix = PipelineFix(
            issue="Test issue",
            fix_type="filter_files",
            details={},
            affected_files=1,
            confidence=0.5,
        )
        assert fix.fix_type == "filter_files"

    def test_pipeline_fix_accept(self):
        """PipelineFix supports accept fix type."""
        fix = PipelineFix(
            issue="Test issue",
            fix_type="accept",
            details={},
            affected_files=1,
            confidence=0.5,
        )
        assert fix.fix_type == "accept"


class TestPipelineValidatorAgentFlow:
    """Integration tests for the PipelineValidator agent flow."""

    @pytest_asyncio.fixture
    async def validator_context(
        self,
        db_session: AsyncSession,
        mock_connector: AsyncMock,
        sample_pipeline: dict[str, Any],
        sample_file_refs: list[FileReference],
    ) -> dict[str, Any]:
        """Create context for full agent flow."""
        return {
            "pipeline": sample_pipeline,
            "sample_files": sample_file_refs,
            "connector": mock_connector,
            "session": db_session,
        }

    @pytest.mark.asyncio()
    async def test_agent_handles_timeout_gracefully(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """PipelineValidator returns partial result on timeout."""
        import asyncio

        async def hang(*args, **kwargs):  # noqa: ARG001
            del args, kwargs
            await asyncio.sleep(1000)

        mock_llm_provider.generate = hang

        agent = PipelineValidator(mock_llm_provider, validator_context)
        # Use type: ignore since we're testing timeout behavior with short duration
        agent.TIMEOUT_SECONDS = 1  # type: ignore[assignment]

        result = await agent.run()

        assert result.success is False
        assert "Timed out" in result.summary

    @pytest.mark.asyncio()
    async def test_agent_handles_llm_error(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """PipelineValidator handles LLM errors gracefully."""
        mock_llm_provider.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        agent = PipelineValidator(mock_llm_provider, validator_context)

        result = await agent.run()

        assert result.success is False
        assert "error" in result.summary.lower()

    def test_agent_tools_are_initialized(
        self,
        mock_llm_provider: AsyncMock,
        validator_context: dict[str, Any],
    ):
        """PipelineValidator initializes all expected tools."""
        agent = PipelineValidator(mock_llm_provider, validator_context)

        expected_tools = [
            "run_dry_run",
            "get_failure_details",
            "try_alternative_config",
            "compare_parser_output",
            "inspect_chunks",
        ]

        for tool_name in expected_tools:
            assert tool_name in agent.tools, f"Missing tool: {tool_name}"
