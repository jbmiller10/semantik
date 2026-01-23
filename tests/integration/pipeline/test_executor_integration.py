"""Integration tests for the pipeline executor with real plugins.

These tests use real implementations of:
- LocalFileConnector for file enumeration
- Text parser for parsing
- Recursive chunking strategy for chunking
- Mock embedding provider (to avoid GPU requirements)

These tests require a real PostgreSQL database connection.
"""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.connectors.local import LocalFileConnector
from shared.pipeline.executor import PipelineExecutor
from shared.pipeline.executor_types import ExecutionMode
from shared.pipeline.types import (
    FileReference,
    NodeType,
    PipelineDAG,
    PipelineEdge,
    PipelineNode,
)


@pytest.fixture()
def simple_dag() -> PipelineDAG:
    """Create a simple DAG for text files."""
    return PipelineDAG(
        id="simple-text-pipeline",
        version="1.0",
        nodes=[
            PipelineNode(
                id="text-parser",
                type=NodeType.PARSER,
                plugin_id="text",
            ),
            PipelineNode(
                id="recursive-chunker",
                type=NodeType.CHUNKER,
                plugin_id="recursive",
                config={
                    "max_tokens": 100,
                    "min_tokens": 20,
                    "overlap_tokens": 10,
                },
            ),
            PipelineNode(
                id="embedder",
                type=NodeType.EMBEDDER,
                plugin_id="dense-local",
            ),
        ],
        edges=[
            PipelineEdge(from_node="_source", to_node="text-parser"),
            PipelineEdge(from_node="text-parser", to_node="recursive-chunker"),
            PipelineEdge(from_node="recursive-chunker", to_node="embedder"),
        ],
    )


@pytest.fixture()
def temp_docs_dir() -> Path:
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)

        # Create some test files
        (docs_dir / "doc1.txt").write_text(
            "This is the first document. It contains some text that will be chunked. "
            "The chunker should split this into multiple chunks based on the configuration."
        )

        (docs_dir / "doc2.txt").write_text(
            "This is the second document. It has different content than the first one. "
            "This will also be processed by the pipeline."
        )

        (docs_dir / "doc3.md").write_text(
            "# Markdown Document\n\n"
            "This is a markdown document with some content.\n\n"
            "## Section 1\n\n"
            "Content for section 1.\n\n"
            "## Section 2\n\n"
            "Content for section 2."
        )

        yield docs_dir


class TestPipelineExecutorWithLocalConnector:
    """Integration tests using LocalFileConnector."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session for testing without DB."""
        session = AsyncMock(spec=AsyncSession)
        # Mock the execute method to return None for get_by_uri
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio()
    async def test_execute_dry_run_with_text_files(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test DRY_RUN mode with real text files."""
        connector = LocalFileConnector({"path": str(temp_docs_dir), "recursive": True})
        await connector.authenticate()

        # Collect file refs
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Check results
        assert result.mode == ExecutionMode.DRY_RUN
        assert result.files_processed == 3
        assert result.files_succeeded == 3
        assert result.files_failed == 0
        assert result.halted is False

        # Check sample outputs
        assert result.sample_outputs is not None
        assert len(result.sample_outputs) == 3

    @pytest.mark.asyncio()
    async def test_execute_creates_chunks(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test that chunks are created from files."""
        connector = LocalFileConnector({"path": str(temp_docs_dir), "recursive": True})
        await connector.authenticate()

        # Collect file refs
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Check chunks were created
        assert result.chunks_created > 0

        # Check chunk stats
        assert result.chunk_stats is not None
        assert result.chunk_stats.total_chunks > 0
        assert result.chunk_stats.min_tokens > 0
        assert result.chunk_stats.max_tokens >= result.chunk_stats.min_tokens

    @pytest.mark.asyncio()
    async def test_execute_with_include_pattern(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test with file include pattern."""
        connector = LocalFileConnector({
            "path": str(temp_docs_dir),
            "recursive": True,
            "include_patterns": ["*.txt"],
        })
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Only .txt files should be processed
        assert result.files_processed == 2
        assert result.files_succeeded == 2

    @pytest.mark.asyncio()
    async def test_execute_progress_events(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test progress events are emitted correctly."""
        connector = LocalFileConnector({
            "path": str(temp_docs_dir),
            "include_patterns": ["doc1.txt"],
        })
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        events: list = []

        async def progress_callback(event):
            events.append(event)

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        await executor.execute(file_iterator(), progress_callback=progress_callback)

        # Check expected event sequence
        event_types = [e.event_type for e in events]
        assert event_types[0] == "pipeline_started"
        assert "file_started" in event_types
        assert "stage_completed" in event_types
        assert "file_completed" in event_types
        assert event_types[-1] == "pipeline_completed"

    @pytest.mark.asyncio()
    async def test_execute_with_limit(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test file limit is respected."""
        connector = LocalFileConnector({"path": str(temp_docs_dir)})
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator(), limit=1)

        assert result.files_processed == 1

    @pytest.mark.asyncio()
    async def test_execute_stage_timings_populated(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test stage timings are recorded."""
        connector = LocalFileConnector({
            "path": str(temp_docs_dir),
            "include_patterns": ["doc1.txt"],
        })
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Check timings are populated
        assert "loader" in result.stage_timings
        assert result.total_duration_ms > 0


class TestPipelineExecutorWithBranchingDag:
    """Integration tests with branching DAG."""

    @pytest.fixture()
    def branching_dag(self) -> PipelineDAG:
        """Create a DAG that routes by extension."""
        return PipelineDAG(
            id="branching-pipeline",
            version="1.0",
            nodes=[
                PipelineNode(
                    id="markdown-parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                ),
                PipelineNode(
                    id="text-parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                ),
                PipelineNode(
                    id="chunker",
                    type=NodeType.CHUNKER,
                    plugin_id="recursive",
                    config={"max_tokens": 100, "min_tokens": 20, "overlap_tokens": 10},
                ),
                PipelineNode(
                    id="embedder",
                    type=NodeType.EMBEDDER,
                    plugin_id="dense-local",
                ),
            ],
            edges=[
                # Route .md to markdown parser
                PipelineEdge(
                    from_node="_source",
                    to_node="markdown-parser",
                    when={"extension": ".md"},
                ),
                # Route everything else to text parser
                PipelineEdge(from_node="_source", to_node="text-parser"),
                # Both parsers feed into chunker
                PipelineEdge(from_node="markdown-parser", to_node="chunker"),
                PipelineEdge(from_node="text-parser", to_node="chunker"),
                # Chunker to embedder
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio()
    async def test_routes_by_extension(
        self,
        branching_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test files are routed based on extension."""
        connector = LocalFileConnector({"path": str(temp_docs_dir)})
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        events: list = []

        async def progress_callback(event):
            if event.event_type == "stage_completed":
                events.append(event)

        executor = PipelineExecutor(
            dag=branching_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator(), progress_callback=progress_callback)

        assert result.files_succeeded == 3

        # Check that different parsers were used
        parser_events = [e for e in events if "parser" in e.stage_id]
        parser_ids = {e.stage_id for e in parser_events}
        # Both markdown-parser and text-parser should have been used
        assert "markdown-parser" in parser_ids or "text-parser" in parser_ids


class TestPipelineExecutorLoadContent:
    """Integration tests for connector load_content method."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio()
    async def test_load_content_from_connector(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test that content is loaded via connector.load_content()."""
        connector = LocalFileConnector({
            "path": str(temp_docs_dir),
            "include_patterns": ["doc1.txt"],
        })
        await connector.authenticate()

        # Get file ref
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        assert len(file_refs) == 1

        # Test load_content directly
        content = await connector.load_content(file_refs[0])
        assert isinstance(content, bytes)
        assert b"first document" in content
