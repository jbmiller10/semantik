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
    async def test_execute_aggregates_stage_timings_across_files(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test that stage timings aggregate across multi-file runs (not last-file only)."""
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
            for i in range(2)
        ]

        async def file_iterator() -> AsyncIterator[FileReference]:
            for f in files:
                yield f

        def fake_record_timing(self: PipelineExecutor, stage_key: str, start_time: float) -> None:  # noqa: ARG001
            self._stage_timings[stage_key] = self._stage_timings.get(stage_key, 0.0) + 1.0

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
            patch.object(PipelineExecutor, "_record_timing", new=fake_record_timing),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        assert result.stage_timings["loader"] == 2.0

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

    @pytest.mark.asyncio()
    async def test_sniff_failure_records_error_in_metadata(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_parser: MagicMock,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test that sniff failures are recorded in file_ref.metadata['errors']."""
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

        # Create a mock sniffer that raises an exception on sniff
        # but uses the real enrich_file_ref so errors propagate to metadata
        from shared.pipeline.sniff import ContentSniffer

        mock_sniffer = MagicMock()
        mock_sniffer.sniff = AsyncMock(side_effect=RuntimeError("Sniff operation failed"))
        mock_sniffer.enrich_file_ref = ContentSniffer().enrich_file_ref

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
            patch(
                "shared.pipeline.executor.ContentSniffer",
                return_value=mock_sniffer,
            ),
        ):
            executor = PipelineExecutor(
                dag=valid_dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor.execute(file_iterator())

        # Processing should still succeed (sniff is non-fatal)
        assert result.files_processed == 1
        assert result.files_succeeded == 1
        assert result.files_failed == 0

        # Check that the sample output has the error recorded in metadata
        assert result.sample_outputs is not None
        assert len(result.sample_outputs) == 1
        sample_file_ref = result.sample_outputs[0].file_ref
        assert "errors" in sample_file_ref.metadata
        assert "sniff" in sample_file_ref.metadata["errors"]
        # Error is now a list from SniffResult.errors
        sniff_errors = sample_file_ref.metadata["errors"]["sniff"]
        assert isinstance(sniff_errors, list)
        assert any("Sniff operation failed" in err for err in sniff_errors)

    @pytest.mark.asyncio()
    async def test_process_file_mid_pipeline_parallel_fanout_executes_all_branches(
        self,
        mock_session: AsyncMock,
        temp_file: Path,
    ) -> None:
        """Test that mid-pipeline parallel fan-out creates and executes multiple paths."""
        dag = PipelineDAG(
            id="test-mid-pipeline-fanout",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="chunker_a", type=NodeType.CHUNKER, plugin_id="chunker_a"),
                PipelineNode(id="chunker_b", type=NodeType.CHUNKER, plugin_id="chunker_b"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-local"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker_a", parallel=True, path_name="path_a"),
                PipelineEdge(from_node="parser", to_node="chunker_b", parallel=True, path_name="path_b"),
                PipelineEdge(from_node="chunker_a", to_node="embedder"),
                PipelineEdge(from_node="chunker_b", to_node="embedder"),
            ],
        )

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        mock_parse_result = MagicMock()
        mock_parse_result.text = "Parsed text content"
        mock_parse_result.metadata = {}
        mock_parser = MagicMock()
        mock_parser.parse_bytes.return_value = mock_parse_result

        def _mk_chunk(chunk_id: str, content: str) -> MagicMock:
            chunk = MagicMock()
            chunk.content = content
            chunk.metadata = MagicMock()
            chunk.metadata.chunk_id = chunk_id
            chunk.metadata.chunk_index = 0
            chunk.metadata.start_offset = 0
            chunk.metadata.end_offset = 10
            chunk.metadata.token_count = 5
            chunk.metadata.hierarchy_level = 0
            return chunk

        strategy_a = MagicMock()
        strategy_a.chunk.return_value = [_mk_chunk("a1", "A chunk")]

        strategy_b = MagicMock()
        strategy_b.chunk.return_value = [_mk_chunk("b1", "B chunk 1"), _mk_chunk("b2", "B chunk 2")]

        def create_strategy_side_effect(plugin_id: str) -> MagicMock:
            if plugin_id == "chunker_a":
                return strategy_a
            if plugin_id == "chunker_b":
                return strategy_b
            raise AssertionError(f"Unexpected chunker plugin_id: {plugin_id}")

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", return_value=mock_parser),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                side_effect=create_strategy_side_effect,
            ),
        ):
            executor = PipelineExecutor(
                dag=dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor._process_file(file_ref, progress_callback=None)

        assert result["skipped"] is False
        assert result["chunks_created"] == 3
        assert "sample_outputs" in result, "expected multiple sample outputs for fan-out"
        sample_outputs = result["sample_outputs"]
        assert {o.path_id for o in sample_outputs} == {"path_a", "path_b"}
        for output in sample_outputs:
            assert all(chunk["path_id"] == output.path_id for chunk in output.chunks)

        assert strategy_a.chunk.call_count == 1
        assert strategy_b.chunk.call_count == 1

    @pytest.mark.asyncio()
    async def test_process_file_does_not_abort_remaining_paths_on_first_failure(
        self,
        mock_session: AsyncMock,
        temp_file: Path,
        mock_chunks: list[MagicMock],
    ) -> None:
        """Test that one path failing doesn't prevent other paths for the same file from running."""
        dag = PipelineDAG(
            id="test-entry-parallel-path-failure",
            version="1.0",
            nodes=[
                PipelineNode(id="parser_fail", type=NodeType.PARSER, plugin_id="fail"),
                PipelineNode(id="parser_ok", type=NodeType.PARSER, plugin_id="ok"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-local"),
            ],
            edges=[
                PipelineEdge(
                    from_node="_source",
                    to_node="parser_fail",
                    when={"extension": ".txt"},
                    parallel=True,
                    path_name="fail_path",
                ),
                PipelineEdge(from_node="_source", to_node="parser_ok", when=None, parallel=False, path_name="ok_path"),
                PipelineEdge(from_node="parser_fail", to_node="chunker"),
                PipelineEdge(from_node="parser_ok", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            extension=".txt",
            size_bytes=100,
            metadata={"source": {"local_path": str(temp_file)}},
        )

        failing_parser = MagicMock()
        failing_parser.parse_bytes.side_effect = RuntimeError("Parse failed")

        ok_parse_result = MagicMock()
        ok_parse_result.text = "Parsed text content"
        ok_parse_result.metadata = {}
        ok_parser = MagicMock()
        ok_parser.parse_bytes.return_value = ok_parse_result

        def get_parser_side_effect(plugin_id: str, _config: dict) -> MagicMock:
            if plugin_id == "fail":
                return failing_parser
            if plugin_id == "ok":
                return ok_parser
            raise AssertionError(f"Unexpected parser plugin_id: {plugin_id}")

        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = mock_chunks

        with (
            patch("shared.plugins.plugin_registry.get", return_value=None),
            patch("shared.pipeline.executor.get_parser", side_effect=get_parser_side_effect),
            patch(
                "shared.pipeline.executor.UnifiedChunkingFactory.create_strategy",
                return_value=mock_strategy,
            ),
        ):
            executor = PipelineExecutor(
                dag=dag,
                collection_id="test-collection",
                session=mock_session,
                mode=ExecutionMode.DRY_RUN,
            )

            result = await executor._process_file(file_ref, progress_callback=None)

        assert result["skipped"] is False
        assert result["chunks_created"] == len(mock_chunks)
        assert result["failed_paths"] == [{"path_id": "fail_path", "error": "Parse failed"}]
        assert "sample_output" in result
        assert result["sample_output"].path_id == "ok_path"

        assert failing_parser.parse_bytes.call_count == 1
        assert ok_parser.parse_bytes.call_count == 1


class TestDatabaseErrorHandling:
    """Tests for database error handling in executor methods."""

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
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio()
    async def test_should_skip_database_error_annotated(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
    ) -> None:
        """Test database errors in _should_skip include stage metadata."""
        executor = PipelineExecutor(
            dag=valid_dag,
            collection_id="test-collection",
            session=mock_session,
            mode=ExecutionMode.FULL,
        )

        # Mock doc_repo to raise an exception
        mock_doc_repo = AsyncMock()
        db_error = Exception("Database connection failed")
        mock_doc_repo.get_by_uri.side_effect = db_error
        executor._doc_repo = mock_doc_repo

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={},
        )

        with pytest.raises(Exception, match="Database connection failed") as exc_info:
            await executor._should_skip(file_ref, "abc123hash")

        # Verify exception has stage context attributes
        assert hasattr(exc_info.value, "stage_id")
        assert exc_info.value.stage_id == "skip_check"  # type: ignore[attr-defined]
        assert hasattr(exc_info.value, "stage_type")
        assert exc_info.value.stage_type == "database"  # type: ignore[attr-defined]

    @pytest.mark.asyncio()
    async def test_create_document_database_error_logged(
        self,
        valid_dag: PipelineDAG,
        mock_session: AsyncMock,
    ) -> None:
        """Test database errors in _create_document are logged and re-raised."""
        executor = PipelineExecutor(
            dag=valid_dag,
            collection_id="test-collection",
            session=mock_session,
            mode=ExecutionMode.FULL,
        )

        # Mock doc_repo to raise an exception
        mock_doc_repo = AsyncMock()
        db_error = Exception("Database insert failed")
        mock_doc_repo.create.side_effect = db_error
        executor._doc_repo = mock_doc_repo

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
            metadata={"source": {"local_path": "/test.txt"}},
        )

        with (
            patch("shared.pipeline.executor.logger") as mock_logger,
            pytest.raises(Exception, match="Database insert failed"),
        ):
            await executor._create_document(file_ref, "abc123hash")

        # Verify error was logged with file URI
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "test.txt" in str(call_args)
        assert "Database insert failed" in str(call_args)
