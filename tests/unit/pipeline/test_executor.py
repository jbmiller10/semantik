"""Unit tests for pipeline executor with mocked plugins."""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.models import DocumentStatus
from shared.pipeline.executor import PipelineExecutor
from shared.pipeline.executor_types import ExecutionMode
from shared.pipeline.types import FileReference, NodeType, PipelineDAG, PipelineEdge, PipelineNode


class TestPipelineExecutorInit:
    """Tests for PipelineExecutor initialization."""

    @pytest.fixture()
    def valid_dag(self) -> PipelineDAG:
        """Create a valid minimal DAG."""
        return PipelineDAG(
            id="test",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-local"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def invalid_dag(self) -> PipelineDAG:
        """Create an invalid DAG (no embedder)."""
        return PipelineDAG(
            id="invalid",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
            ],
        )

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock()

    def test_init_with_valid_dag(self, valid_dag: PipelineDAG, mock_session: AsyncMock) -> None:
        """Test initialization with valid DAG."""
        executor = PipelineExecutor(
            dag=valid_dag,
            collection_id="test-collection",
            session=mock_session,
        )

        assert executor.dag == valid_dag
        assert executor.collection_id == "test-collection"
        assert executor.mode == ExecutionMode.FULL

    def test_init_with_dry_run_mode(self, valid_dag: PipelineDAG, mock_session: AsyncMock) -> None:
        """Test initialization with DRY_RUN mode."""
        executor = PipelineExecutor(
            dag=valid_dag,
            collection_id="test-collection",
            session=mock_session,
            mode=ExecutionMode.DRY_RUN,
        )

        assert executor.mode == ExecutionMode.DRY_RUN

    def test_init_with_invalid_dag_raises(self, invalid_dag: PipelineDAG, mock_session: AsyncMock) -> None:
        """Test initialization with invalid DAG raises ValueError."""
        with pytest.raises(ValueError, match="Invalid DAG"):
            PipelineExecutor(
                dag=invalid_dag,
                collection_id="test-collection",
                session=mock_session,
            )


class TestPipelineExecutorExecute:
    """Tests for PipelineExecutor.execute() method."""

    @pytest.fixture()
    def valid_dag(self) -> PipelineDAG:
        """Create a valid DAG for testing."""
        return PipelineDAG(
            id="test",
            version="1.0",
            nodes=[
                PipelineNode(
                    id="parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                ),
                PipelineNode(
                    id="chunker",
                    type=NodeType.CHUNKER,
                    plugin_id="recursive",
                    config={"max_tokens": 500, "min_tokens": 50, "overlap_tokens": 25},
                ),
                PipelineNode(
                    id="embedder",
                    type=NodeType.EMBEDDER,
                    plugin_id="dense-local",
                ),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def temp_file(self) -> Path:
        """Create a temp file with content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Hello, World! This is a test document.")
            return Path(f.name)

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        return session

    @pytest.fixture()
    def mock_parser(self) -> MagicMock:
        """Create a mock parser."""
        mock_result = MagicMock()
        mock_result.text = "Parsed text content"
        mock_result.metadata = {"pages": 1}

        parser = MagicMock()
        parser.parse_bytes.return_value = mock_result
        return parser

    @pytest.fixture()
    def mock_chunks(self) -> list[MagicMock]:
        """Create mock chunk objects."""
        chunks = []
        for i in range(3):
            chunk = MagicMock()
            chunk.content = f"Chunk {i} content"
            chunk.metadata = MagicMock()
            chunk.metadata.chunk_id = f"chunk_{i}"
            chunk.metadata.chunk_index = i
            chunk.metadata.start_offset = i * 10
            chunk.metadata.end_offset = (i + 1) * 10
            chunk.metadata.token_count = 50
            chunk.metadata.hierarchy_level = 0
            chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio()
    async def test_execute_empty_iterator(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
    ) -> None:
        """Test executing with empty file iterator."""
        executor = PipelineExecutor(
            dag=valid_dag,
            collection_id="test-collection",
            session=mock_session,
        )

        async def empty_iterator() -> AsyncIterator[FileReference]:
            return
            yield  # Make it a generator

        result = await executor.execute(empty_iterator())

        assert result.files_processed == 0
        assert result.files_succeeded == 0
        assert result.files_failed == 0
        assert result.halted is False

    @pytest.mark.asyncio()
    async def test_execute_with_limit(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test executing with file limit."""
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        # Create multiple file refs
        files = [
            FileReference(
                uri=f"file:///test{i}.txt",
                source_type="directory",
                content_type="document",
                size_bytes=100,
                metadata={"source": {"local_path": str(temp_file)}},
            )
            for i in range(5)
        ]

        async def file_iterator() -> AsyncIterator[FileReference]:
            for f in files:
                yield f

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator(), limit=2)

        assert result.files_processed == 2
        assert result.files_succeeded == 2

    @pytest.mark.asyncio()
    async def test_execute_records_stage_timings(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test that stage timings are recorded."""
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        assert "loader" in result.stage_timings
        assert result.stage_timings["loader"] > 0

    @pytest.mark.asyncio()
    async def test_execute_dry_run_returns_samples(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test DRY_RUN mode returns sample outputs."""
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        assert result.sample_outputs is not None
        assert len(result.sample_outputs) == 1
        assert result.sample_outputs[0].file_ref.uri == "file:///test.txt"

    @pytest.mark.asyncio()
    async def test_execute_progress_callback(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test progress callback is called."""
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        events: list = []

        async def progress_callback(event):
            events.append(event)

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            await executor.execute(file_iterator(), progress_callback=progress_callback)

        # Check we got expected events
        event_types = [e.event_type for e in events]
        assert "pipeline_started" in event_types
        assert "file_started" in event_types
        assert "file_completed" in event_types
        assert "pipeline_completed" in event_types

    @pytest.mark.asyncio()
    async def test_execute_handles_parser_failure(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
    ) -> None:
        """Test that parser failures are recorded."""
        mock_parser = MagicMock()
        mock_parser.parse_bytes.side_effect = Exception("Parse failed")

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        # Patch both the plugin registry (new path) and legacy get_parser (fallback)
        # to ensure the mock parser is used
        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        assert result.files_failed == 1
        assert len(result.failures) == 1
        assert "Parse failed" in result.failures[0].error_message

    @pytest.mark.asyncio()
    async def test_execute_halts_on_consecutive_failures(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
    ) -> None:
        """Test pipeline halts after consecutive failures."""
        mock_parser = MagicMock()
        mock_parser.parse_bytes.side_effect = Exception("Parse failed")

        files = [
            FileReference(
                uri=f"file:///test{i}.txt",
                source_type="directory",
                content_type="document",
                size_bytes=100,
                metadata={"source": {"local_path": str(temp_file)}},
            )
            for i in range(20)
        ]

        async def file_iterator() -> AsyncIterator[FileReference]:
            for f in files:
                yield f

        # Patch both the plugin registry (new path) and legacy get_parser (fallback)
        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
                consecutive_failure_threshold=5,
            )

            result = await executor.execute(file_iterator())

        # Should have halted after 5 consecutive failures
        assert result.halted is True
        assert result.files_processed < 20
        assert result.halt_reason is not None

    @pytest.mark.asyncio()
    async def test_execute_continues_after_single_failure(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test pipeline continues after a single failure."""
        call_count = 0

        def parser_side_effect(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Parse failed for file 2")
            result = MagicMock()
            result.text = "Parsed text"
            result.metadata = {}
            return result

        mock_parser = MagicMock()
        mock_parser.parse_bytes.side_effect = parser_side_effect

        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        files = [
            FileReference(
                uri=f"file:///test{i}.txt",
                source_type="directory",
                content_type="document",
                size_bytes=100,
                metadata={"source": {"local_path": str(temp_file)}},
            )
            for i in range(3)
        ]

        async def file_iterator() -> AsyncIterator[FileReference]:
            for f in files:
                yield f

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        assert result.files_processed == 3
        assert result.files_succeeded == 2
        assert result.files_failed == 1
        assert result.halted is False

    @pytest.mark.asyncio()
    async def test_execute_calculates_chunk_stats(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
    ) -> None:
        """Test chunk statistics are calculated."""
        # Create chunks with varying token counts
        chunks = []
        for i, tokens in enumerate([100, 200, 300]):
            chunk = MagicMock()
            chunk.content = f"Chunk {i}"
            chunk.metadata = MagicMock()
            chunk.metadata.chunk_id = f"chunk_{i}"
            chunk.metadata.chunk_index = i
            chunk.metadata.start_offset = 0
            chunk.metadata.end_offset = 10
            chunk.metadata.token_count = tokens
            chunk.metadata.hierarchy_level = 0
            chunks.append(chunk)

        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = chunks

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        assert result.chunk_stats is not None
        assert result.chunk_stats.total_chunks == 3
        assert result.chunk_stats.avg_tokens == 200.0
        assert result.chunk_stats.min_tokens == 100
        assert result.chunk_stats.max_tokens == 300

    @pytest.mark.asyncio()
    async def test_should_skip_unchanged_document_in_full_mode(
        self,
        valid_dag: PipelineDAG,
        temp_file: Path,
    ) -> None:
        """Test that unchanged documents are skipped in FULL mode."""
        # Create mock session
        mock_session = AsyncMock()

        # Create a mock existing document with matching content hash
        mock_existing_doc = MagicMock()
        mock_existing_doc.content_hash = (
            "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"  # SHA256 of "Hello World"
        )
        mock_existing_doc.status = DocumentStatus.COMPLETED

        # Create a mock doc repo that returns the existing document
        mock_doc_repo = MagicMock()
        mock_doc_repo.get_by_uri = AsyncMock(return_value=mock_existing_doc)

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=11,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        # Write "Hello World" to temp file to match the hash
        temp_file.write_text("Hello World")

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        with patch(
            "shared.pipeline.executor.DocumentRepository",
            return_value=mock_doc_repo,
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.FULL,  # FULL mode - change detection enabled
            )

            result = await executor.execute(file_iterator())

        # Should be skipped due to unchanged content
        assert result.files_processed == 1
        assert result.files_skipped == 1
        assert result.files_succeeded == 0
        assert result.files_failed == 0

    @pytest.mark.asyncio()
    async def test_failing_callback_does_not_crash_pipeline(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test that a failing progress callback doesn't stop execution."""
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        async def file_iterator() -> AsyncIterator[FileReference]:
            yield file_ref

        callback_call_count = 0

        async def failing_callback(event):  # noqa: ARG001
            nonlocal callback_call_count
            callback_call_count += 1
            raise RuntimeError("Callback failed!")

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            # Execute should complete despite failing callback
            result = await executor.execute(file_iterator(), progress_callback=failing_callback)

        # Pipeline should complete successfully
        assert result.files_processed == 1
        assert result.files_succeeded == 1
        assert result.files_failed == 0
        # Callback should have been called multiple times (events still emitted)
        assert callback_call_count > 0
