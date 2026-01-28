"""Unit tests for pipeline validation sub-agent tools.

These tests verify the tools used by the PipelineValidator sub-agent
for testing and validating pipeline configurations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.pipeline.executor_types import ChunkStats, ExecutionMode, ExecutionResult, SampleOutput, StageFailure

# StageFailure is used in tests directly for creating test data
from shared.pipeline.types import FileReference
from webui.services.agent.tools.subagent_tools.validation import (
    CompareParserOutputTool,
    GetFailureDetailsTool,
    InspectChunksTool,
    RunDryRunTool,
    TryAlternativeConfigTool,
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
            {
                "id": "parser",
                "type": "parser",
                "plugin_id": "text",
                "config": {},
            },
            {
                "id": "chunker",
                "type": "chunker",
                "plugin_id": "character",
                "config": {"max_tokens": 1000},
            },
            {
                "id": "embedder",
                "type": "embedder",
                "plugin_id": "mock",
                "config": {},
            },
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
    connector.load_content = AsyncMock(return_value=b"Sample content for testing")
    return connector


@pytest.fixture()
def mock_session() -> AsyncMock:
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture()
def successful_execution_result(sample_file_refs: list[FileReference]) -> ExecutionResult:
    """Create a successful execution result."""
    return ExecutionResult(
        mode=ExecutionMode.DRY_RUN,
        files_processed=2,
        files_succeeded=2,
        files_failed=0,
        files_skipped=0,
        chunks_created=10,
        chunk_stats=ChunkStats(
            total_chunks=10,
            avg_tokens=200.0,
            min_tokens=100,
            max_tokens=300,
        ),
        failures=[],
        stage_timings={"parser:parser": 100.0, "chunker:chunker": 50.0},
        total_duration_ms=200.0,
        sample_outputs=[
            SampleOutput(
                file_ref=sample_file_refs[0],
                chunks=[
                    {"content": "Chunk 1 content here", "metadata": {"token_count": 150}},
                    {"content": "Chunk 2 content here", "metadata": {"token_count": 200}},
                ],
                parse_metadata={"pages": 5},
            ),
            SampleOutput(
                file_ref=sample_file_refs[1],
                chunks=[
                    {"content": "Markdown chunk", "metadata": {"token_count": 100}},
                ],
                parse_metadata={},
            ),
        ],
    )


@pytest.fixture()
def failed_execution_result(sample_file_refs: list[FileReference]) -> ExecutionResult:
    """Create an execution result with failures."""
    return ExecutionResult(
        mode=ExecutionMode.DRY_RUN,
        files_processed=2,
        files_succeeded=1,
        files_failed=1,
        files_skipped=0,
        chunks_created=5,
        chunk_stats=ChunkStats(
            total_chunks=5,
            avg_tokens=200.0,
            min_tokens=100,
            max_tokens=300,
        ),
        failures=[
            StageFailure(
                file_uri="file:///docs/paper1.pdf",
                stage_id="parser",
                stage_type="parser",
                error_type="ParseError",
                error_message="Failed to parse PDF: corrupted file",
                error_traceback="Traceback...",
            ),
        ],
        stage_timings={"parser:parser": 100.0},
        total_duration_ms=150.0,
        sample_outputs=[
            SampleOutput(
                file_ref=sample_file_refs[1],
                chunks=[{"content": "Markdown chunk", "metadata": {"token_count": 100}}],
                parse_metadata={},
            ),
        ],
    )


# RunDryRunTool tests
class TestRunDryRunTool:
    """Tests for the RunDryRunTool."""

    @pytest.mark.asyncio()
    async def test_dry_run_returns_correct_stats(
        self,
        mock_connector: AsyncMock,
        mock_session: AsyncMock,
        sample_file_refs: list[FileReference],
        sample_pipeline: dict[str, Any],
        successful_execution_result: ExecutionResult,
    ):
        """Test that dry run returns correct statistics."""
        context: dict[str, Any] = {
            "connector": mock_connector,
            "session": mock_session,
            "sample_files": sample_file_refs,
        }
        tool = RunDryRunTool(context)

        with (
            patch("shared.pipeline.executor.PipelineExecutor") as mock_executor_class,
            patch("shared.plugins.registry.plugin_registry") as mock_registry,
        ):
            # Mock the plugin IDs that match the sample pipeline (text, character, mock)
            mock_registry.list_ids.return_value = ["text", "character", "mock"]
            mock_executor = AsyncMock()
            mock_executor.execute = AsyncMock(return_value=successful_execution_result)
            mock_executor_class.return_value = mock_executor

            result = await tool.execute(pipeline=sample_pipeline, sample_count=50)

        assert result["success"] is True
        assert result["files_tested"] == 2
        assert result["files_succeeded"] == 2
        assert result["files_failed"] == 0
        assert result["success_rate"] == 1.0
        assert result["assessment"] == "ready"

    @pytest.mark.asyncio()
    async def test_dry_run_calculates_success_rate(
        self,
        mock_connector: AsyncMock,
        mock_session: AsyncMock,
        sample_file_refs: list[FileReference],
        sample_pipeline: dict[str, Any],
        failed_execution_result: ExecutionResult,
    ):
        """Test that success rate is calculated correctly."""
        context: dict[str, Any] = {
            "connector": mock_connector,
            "session": mock_session,
            "sample_files": sample_file_refs,
        }
        tool = RunDryRunTool(context)

        with (
            patch("shared.pipeline.executor.PipelineExecutor") as mock_executor_class,
            patch("shared.plugins.registry.plugin_registry") as mock_registry,
        ):
            # Mock the plugin IDs that match the sample pipeline (text, character, mock)
            mock_registry.list_ids.return_value = ["text", "character", "mock"]
            mock_executor = AsyncMock()
            mock_executor.execute = AsyncMock(return_value=failed_execution_result)
            mock_executor_class.return_value = mock_executor

            result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is True
        assert result["success_rate"] == 0.5  # 1 of 2 succeeded
        assert result["assessment"] == "blocking_issues"  # <90%

    @pytest.mark.asyncio()
    async def test_dry_run_stores_results_in_context(
        self,
        mock_connector: AsyncMock,
        mock_session: AsyncMock,
        sample_file_refs: list[FileReference],
        sample_pipeline: dict[str, Any],
        failed_execution_result: ExecutionResult,
    ):
        """Test that results are stored in context for other tools."""
        context: dict[str, Any] = {
            "connector": mock_connector,
            "session": mock_session,
            "sample_files": sample_file_refs,
        }
        tool = RunDryRunTool(context)

        with (
            patch("shared.pipeline.executor.PipelineExecutor") as mock_executor_class,
            patch("shared.plugins.registry.plugin_registry") as mock_registry,
        ):
            # Mock the plugin IDs that match the sample pipeline (text, character, mock)
            mock_registry.list_ids.return_value = ["text", "character", "mock"]
            mock_executor = AsyncMock()
            mock_executor.execute = AsyncMock(return_value=failed_execution_result)
            mock_executor_class.return_value = mock_executor

            await tool.execute(pipeline=sample_pipeline)

        assert "_dry_run_result" in context
        assert "_dry_run_failures" in context
        assert "file:///docs/paper1.pdf" in context["_dry_run_failures"]

    @pytest.mark.asyncio()
    async def test_dry_run_fails_without_session(self, sample_pipeline: dict[str, Any]):
        """Test that dry run fails without database session."""
        context: dict[str, Any] = {}
        tool = RunDryRunTool(context)

        result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is False
        assert "session" in result["error"]

    @pytest.mark.asyncio()
    async def test_dry_run_fails_without_connector(self, mock_session: AsyncMock, sample_pipeline: dict[str, Any]):
        """Test that dry run fails without connector."""
        context: dict[str, Any] = {"session": mock_session}
        tool = RunDryRunTool(context)

        result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is False
        assert "connector" in result["error"]

    @pytest.mark.asyncio()
    async def test_dry_run_fails_without_sample_files(
        self, mock_session: AsyncMock, mock_connector: AsyncMock, sample_pipeline: dict[str, Any]
    ):
        """Test that dry run fails without sample files."""
        context: dict[str, Any] = {
            "session": mock_session,
            "connector": mock_connector,
        }
        tool = RunDryRunTool(context)

        result = await tool.execute(pipeline=sample_pipeline)

        assert result["success"] is False
        assert "sample files" in result["error"].lower()


# GetFailureDetailsTool tests
class TestGetFailureDetailsTool:
    """Tests for the GetFailureDetailsTool."""

    @pytest.mark.asyncio()
    async def test_get_failure_details_returns_info(self):
        """Test that failure details are returned correctly."""
        failure = StageFailure(
            file_uri="file:///docs/paper1.pdf",
            stage_id="parser",
            stage_type="parser",
            error_type="ParseError",
            error_message="Failed to parse PDF",
            error_traceback="Traceback...",
        )
        context: dict[str, Any] = {
            "_dry_run_failures": {"file:///docs/paper1.pdf": failure},
        }
        tool = GetFailureDetailsTool(context)

        result = await tool.execute(file_uri="file:///docs/paper1.pdf")

        assert result["success"] is True
        assert result["file_uri"] == "file:///docs/paper1.pdf"
        assert result["stage_id"] == "parser"
        assert result["error_type"] == "ParseError"
        assert result["error_message"] == "Failed to parse PDF"

    @pytest.mark.asyncio()
    async def test_get_failure_details_not_found(self):
        """Test that error is returned for unknown file."""
        # Need at least one failure entry to pass the "no dry-run results" check
        dummy_failure = StageFailure(
            file_uri="file:///other.pdf",
            stage_id="parser",
            stage_type="parser",
            error_type="ParseError",
            error_message="Other file error",
            error_traceback="",
        )
        context: dict[str, Any] = {
            "_dry_run_failures": {"file:///other.pdf": dummy_failure},
        }
        tool = GetFailureDetailsTool(context)

        result = await tool.execute(file_uri="file:///unknown.pdf")

        assert result["success"] is False
        assert "no failure found" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_get_failure_details_without_dry_run(self):
        """Test error when dry run hasn't been executed."""
        context: dict[str, Any] = {}
        tool = GetFailureDetailsTool(context)

        result = await tool.execute(file_uri="file:///docs/paper1.pdf")

        assert result["success"] is False
        assert "run_dry_run" in result["error"]


# TryAlternativeConfigTool tests
class TestTryAlternativeConfigTool:
    """Tests for the TryAlternativeConfigTool."""

    @pytest.mark.asyncio()
    async def test_try_alternative_success(
        self,
        mock_connector: AsyncMock,
        mock_session: AsyncMock,
        sample_file_refs: list[FileReference],
    ):
        """Test successful alternative parser trial."""
        context: dict[str, Any] = {
            "connector": mock_connector,
            "session": mock_session,
            "_enumerated_files": sample_file_refs,
            "sample_files": sample_file_refs,
        }
        tool = TryAlternativeConfigTool(context)

        # Create a successful result for the alternative
        success_result = ExecutionResult(
            mode=ExecutionMode.DRY_RUN,
            files_processed=1,
            files_succeeded=1,
            files_failed=0,
            files_skipped=0,
            chunks_created=3,
            chunk_stats=None,
            failures=[],
            stage_timings={},
            total_duration_ms=50.0,
            sample_outputs=[
                SampleOutput(
                    file_ref=sample_file_refs[0],
                    chunks=[{"content": "Parsed content"}],
                    parse_metadata={"method": "alternative"},
                )
            ],
        )

        with (
            patch("shared.pipeline.executor.PipelineExecutor") as mock_executor_class,
            patch("shared.plugins.loader.load_plugins"),
            patch("shared.plugins.plugin_registry") as mock_registry,
        ):
            mock_registry.get.return_value = MagicMock()  # Parser exists
            mock_executor = AsyncMock()
            mock_executor.execute = AsyncMock(return_value=success_result)
            mock_executor_class.return_value = mock_executor

            result = await tool.execute(
                file_uri="file:///docs/paper1.pdf",
                parser_id="unstructured",
            )

        assert result["success"] is True
        assert result["parse_success"] is True
        assert result["parser_id"] == "unstructured"

    @pytest.mark.asyncio()
    async def test_try_alternative_file_not_found(self, mock_session: AsyncMock):
        """Test error when file is not found."""
        context: dict[str, Any] = {
            "session": mock_session,
            "_enumerated_files": [],
            "sample_files": [],
        }
        tool = TryAlternativeConfigTool(context)

        result = await tool.execute(
            file_uri="file:///unknown.pdf",
            parser_id="text",
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_try_alternative_parser_not_found(
        self,
        mock_session: AsyncMock,
        sample_file_refs: list[FileReference],
    ):
        """Test error when parser doesn't exist."""
        context: dict[str, Any] = {
            "session": mock_session,
            "_enumerated_files": sample_file_refs,
            "sample_files": sample_file_refs,
        }
        tool = TryAlternativeConfigTool(context)

        with (
            patch("shared.plugins.loader.load_plugins"),
            patch("shared.plugins.plugin_registry") as mock_registry,
        ):
            mock_registry.get.return_value = None  # Parser doesn't exist
            mock_registry.get_by_type.return_value = {"text": MagicMock()}

            result = await tool.execute(
                file_uri="file:///docs/paper1.pdf",
                parser_id="nonexistent_parser",
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "available_parsers" in result


# CompareParserOutputTool tests
class TestCompareParserOutputTool:
    """Tests for the CompareParserOutputTool."""

    @pytest.mark.asyncio()
    async def test_compare_parsers_success(
        self,
        mock_connector: AsyncMock,
        sample_file_refs: list[FileReference],
    ):
        """Test successful parser comparison."""
        mock_connector.load_content = AsyncMock(return_value=b"# Heading\n\nContent here.")

        context: dict[str, Any] = {
            "connector": mock_connector,
            "_enumerated_files": sample_file_refs,
            "sample_files": sample_file_refs,
        }
        tool = CompareParserOutputTool(context)

        # Create mock parser plugins
        mock_parser_a = MagicMock()
        mock_parser_a.parse_bytes.return_value = MagicMock(
            text="Parsed by A with more content here and tables|", metadata={}
        )
        mock_parser_b = MagicMock()
        mock_parser_b.parse_bytes.return_value = MagicMock(text="Parsed by B", metadata={})

        with (
            patch("shared.plugins.loader.load_plugins"),
            patch("shared.plugins.plugin_registry") as mock_registry,
        ):
            # Return different mocks for each parser
            def get_parser(_plugin_type: str, plugin_id: str) -> MagicMock | None:
                if plugin_id == "parser_a":
                    record = MagicMock()
                    record.plugin_class.return_value = mock_parser_a
                    return record
                if plugin_id == "parser_b":
                    record = MagicMock()
                    record.plugin_class.return_value = mock_parser_b
                    return record
                return None

            mock_registry.get.side_effect = get_parser

            result = await tool.execute(
                file_uri="file:///docs/notes.md",
                parser_a="parser_a",
                parser_b="parser_b",
            )

        assert result["success"] is True
        assert "results" in result
        assert result["parser_a"] == "parser_a"
        assert result["parser_b"] == "parser_b"

    @pytest.mark.asyncio()
    async def test_compare_parsers_one_fails(
        self,
        mock_connector: AsyncMock,
        sample_file_refs: list[FileReference],
    ):
        """Test comparison when one parser fails."""
        mock_connector.load_content = AsyncMock(return_value=b"Content")

        context: dict[str, Any] = {
            "connector": mock_connector,
            "_enumerated_files": sample_file_refs,
            "sample_files": sample_file_refs,
        }
        tool = CompareParserOutputTool(context)

        mock_parser_a = MagicMock()
        mock_parser_a.parse_bytes.return_value = MagicMock(text="Parsed", metadata={})

        with (
            patch("shared.plugins.loader.load_plugins"),
            patch("shared.plugins.plugin_registry") as mock_registry,
        ):

            def get_parser(_plugin_type: str, plugin_id: str) -> MagicMock | None:
                if plugin_id == "parser_a":
                    record = MagicMock()
                    record.plugin_class.return_value = mock_parser_a
                    return record
                if plugin_id == "parser_b":
                    return None  # Parser B not found
                return None

            mock_registry.get.side_effect = get_parser

            result = await tool.execute(
                file_uri="file:///docs/notes.md",
                parser_a="parser_a",
                parser_b="parser_b",
            )

        assert result["success"] is True
        assert result["results"]["parser_a"]["success"] is True
        assert result["results"]["parser_b"]["success"] is False

    @pytest.mark.asyncio()
    async def test_compare_parsers_file_not_found(self, mock_connector: AsyncMock):
        """Test error when file not found."""
        context: dict[str, Any] = {
            "connector": mock_connector,
            "_enumerated_files": [],
            "sample_files": [],
        }
        tool = CompareParserOutputTool(context)

        result = await tool.execute(
            file_uri="file:///unknown.pdf",
            parser_a="text",
            parser_b="unstructured",
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# InspectChunksTool tests
class TestInspectChunksTool:
    """Tests for the InspectChunksTool."""

    @pytest.mark.asyncio()
    async def test_inspect_chunks_returns_stats(self, sample_file_refs: list[FileReference]):
        """Test that chunk statistics are returned correctly."""
        sample_output = SampleOutput(
            file_ref=sample_file_refs[0],
            chunks=[
                {"content": "Chunk 1 content", "metadata": {"token_count": 100}},
                {"content": "Chunk 2 content longer", "metadata": {"token_count": 150}},
                {"content": "Chunk 3", "metadata": {"token_count": 50}},
            ],
            parse_metadata={"pages": 3},
        )
        context: dict[str, Any] = {
            "_sample_outputs": {"file:///docs/paper1.pdf": sample_output},
        }
        tool = InspectChunksTool(context)

        result = await tool.execute(file_uri="file:///docs/paper1.pdf")

        assert result["success"] is True
        assert result["chunk_count"] == 3
        assert "size_stats" in result
        assert "token_stats" in result
        assert result["token_stats"]["avg_tokens"] == 100.0  # (100+150+50)/3
        assert result["token_stats"]["min_tokens"] == 50
        assert result["token_stats"]["max_tokens"] == 150

    @pytest.mark.asyncio()
    async def test_inspect_chunks_includes_previews(self, sample_file_refs: list[FileReference]):
        """Test that chunk previews are included."""
        sample_output = SampleOutput(
            file_ref=sample_file_refs[0],
            chunks=[
                {"content": "Chunk 1 content", "metadata": {}},
                {"content": "Chunk 2 content", "metadata": {}},
            ],
            parse_metadata={},
        )
        context: dict[str, Any] = {
            "_sample_outputs": {"file:///docs/paper1.pdf": sample_output},
        }
        tool = InspectChunksTool(context)

        result = await tool.execute(file_uri="file:///docs/paper1.pdf", preview_count=2)

        assert result["success"] is True
        assert "previews" in result
        assert len(result["previews"]) == 2
        assert result["previews"][0]["preview"] == "Chunk 1 content"

    @pytest.mark.asyncio()
    async def test_inspect_chunks_file_not_found(self, sample_file_refs: list[FileReference]):
        """Test error when file not in sample outputs."""
        # Need at least one sample output entry to pass the "no sample outputs" check
        dummy_sample = SampleOutput(
            file_ref=sample_file_refs[0],
            chunks=[{"content": "Dummy chunk"}],
            parse_metadata={},
        )
        context: dict[str, Any] = {
            "_sample_outputs": {"file:///docs/paper1.pdf": dummy_sample},
        }
        tool = InspectChunksTool(context)

        result = await tool.execute(file_uri="file:///unknown.pdf")

        assert result["success"] is False
        assert "no chunks found" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_inspect_chunks_without_dry_run(self):
        """Test error when dry run hasn't been executed."""
        context: dict[str, Any] = {}
        tool = InspectChunksTool(context)

        result = await tool.execute(file_uri="file:///docs/paper1.pdf")

        assert result["success"] is False
        assert "run_dry_run" in result["error"]


# Tool schema tests
class TestValidationToolSchemas:
    """Tests for validation tool schema generation."""

    def test_run_dry_run_schema(self):
        """Test RunDryRunTool schema."""
        tool = RunDryRunTool({})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "run_dry_run"
        assert "pipeline" in schema["function"]["parameters"]["properties"]
        assert "sample_count" in schema["function"]["parameters"]["properties"]
        assert "pipeline" in schema["function"]["parameters"]["required"]

    def test_get_failure_details_schema(self):
        """Test GetFailureDetailsTool schema."""
        tool = GetFailureDetailsTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "get_failure_details"
        assert "file_uri" in schema["function"]["parameters"]["required"]

    def test_try_alternative_config_schema(self):
        """Test TryAlternativeConfigTool schema."""
        tool = TryAlternativeConfigTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "try_alternative_config"
        assert "file_uri" in schema["function"]["parameters"]["required"]
        assert "parser_id" in schema["function"]["parameters"]["required"]
        assert "parser_config" in schema["function"]["parameters"]["properties"]

    def test_compare_parser_output_schema(self):
        """Test CompareParserOutputTool schema."""
        tool = CompareParserOutputTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "compare_parser_output"
        assert "file_uri" in schema["function"]["parameters"]["required"]
        assert "parser_a" in schema["function"]["parameters"]["required"]
        assert "parser_b" in schema["function"]["parameters"]["required"]

    def test_inspect_chunks_schema(self):
        """Test InspectChunksTool schema."""
        tool = InspectChunksTool({})
        schema = tool.get_schema()

        assert schema["function"]["name"] == "inspect_chunks"
        assert "file_uri" in schema["function"]["parameters"]["required"]
        assert "preview_count" in schema["function"]["parameters"]["properties"]
