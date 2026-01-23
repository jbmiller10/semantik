"""Pipeline executor for processing files through the DAG.

This module provides the PipelineExecutor class that orchestrates file
processing through a pipeline DAG, coordinating parsing, chunking,
extraction, and embedding stages.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.unified.factory import UnifiedChunkingFactory
from shared.database.repositories.document_repository import DocumentRepository
from shared.pipeline.executor_types import (
    ChunkStats,
    ExecutionMode,
    ExecutionResult,
    ProgressEvent,
    SampleOutput,
    StageFailure,
)
from shared.pipeline.failure_tracker import ConsecutiveFailureTracker
from shared.pipeline.loader import LoadError, PipelineLoader
from shared.pipeline.router import PipelineRouter
from shared.pipeline.types import FileReference, NodeType, PipelineDAG, PipelineNode
from shared.text_processing.parsers.registry import get_parser

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Type alias for progress callbacks
ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


class PipelineExecutionError(Exception):
    """Error raised when pipeline execution fails."""

    def __init__(self, message: str, failures: list[StageFailure] | None = None) -> None:
        self.failures = failures or []
        super().__init__(message)


class PipelineExecutor:
    """Executes files through a pipeline DAG.

    The executor coordinates the processing of files through the pipeline,
    handling routing, change detection, failure tracking, and progress
    reporting.

    Example:
        ```python
        executor = PipelineExecutor(
            dag=my_dag,
            collection_id="uuid",
            session=db_session,
            connector=my_connector,
            mode=ExecutionMode.FULL,
        )

        async for file_ref in connector.enumerate():
            pass  # Collect file refs

        result = await executor.execute(file_refs_iterator)
        print(f"Processed {result.files_succeeded} files")
        ```

    Attributes:
        dag: The pipeline DAG to execute
        collection_id: UUID of the target collection
        mode: Execution mode (FULL or DRY_RUN)
    """

    def __init__(
        self,
        dag: PipelineDAG,
        collection_id: str,
        session: AsyncSession,
        connector: BaseConnector | None = None,
        mode: ExecutionMode = ExecutionMode.FULL,
        consecutive_failure_threshold: int = 10,
    ) -> None:
        """Initialize the pipeline executor.

        Args:
            dag: The pipeline DAG definition
            collection_id: UUID of the collection to populate
            session: Async database session
            connector: Optional connector for loading content from non-file:// URIs
            mode: Execution mode (FULL for production, DRY_RUN for validation)
            consecutive_failure_threshold: Number of consecutive failures before halt

        Raises:
            ValueError: If the DAG is invalid
        """
        # Validate DAG on init - fail fast
        errors = dag.validate()
        if errors:
            error_msgs = "; ".join(e.message for e in errors)
            raise ValueError(f"Invalid DAG: {error_msgs}")

        self.dag = dag
        self.collection_id = collection_id
        self.session = session
        self.mode = mode
        self._connector = connector

        # Initialize components
        self._router = PipelineRouter(dag)
        self._loader = PipelineLoader(connector)
        self._failure_tracker = ConsecutiveFailureTracker(threshold=consecutive_failure_threshold)

        # Repositories
        self._doc_repo = DocumentRepository(session)

        # Execution state
        self._stage_timings: dict[str, float] = {}

    async def execute(
        self,
        file_refs: AsyncIterator[FileReference],
        limit: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ExecutionResult:
        """Execute the pipeline over a set of file references.

        Args:
            file_refs: Async iterator of file references to process
            limit: Optional maximum number of files to process
            progress_callback: Optional async callback for progress events

        Returns:
            ExecutionResult with summary statistics and any failures
        """
        start_time = time.perf_counter()

        # Counters
        files_processed = 0
        files_succeeded = 0
        files_failed = 0
        files_skipped = 0
        chunks_created = 0
        all_token_counts: list[int] = []
        failures: list[StageFailure] = []
        sample_outputs: list[SampleOutput] = [] if self.mode == ExecutionMode.DRY_RUN else None  # type: ignore[assignment]

        # Emit pipeline started event
        await self._emit_progress(
            progress_callback,
            ProgressEvent(
                event_type="pipeline_started",
                details={"mode": self.mode.value, "collection_id": self.collection_id},
            ),
        )

        # Process files
        async for file_ref in file_refs:
            # Check limit
            if limit is not None and files_processed >= limit:
                break

            # Check if we should halt due to consecutive failures
            if self._failure_tracker.should_halt():
                logger.warning("Halting pipeline due to consecutive failures")
                await self._emit_progress(
                    progress_callback,
                    ProgressEvent(
                        event_type="pipeline_halted",
                        details={"reason": self._failure_tracker.get_halt_reason()},
                    ),
                )
                break

            files_processed += 1

            # Emit file started event
            await self._emit_progress(
                progress_callback,
                ProgressEvent(event_type="file_started", file_uri=file_ref.uri),
            )

            try:
                # Process the file
                result = await self._process_file(file_ref, progress_callback)

                if result["skipped"]:
                    files_skipped += 1
                    await self._emit_progress(
                        progress_callback,
                        ProgressEvent(
                            event_type="file_skipped",
                            file_uri=file_ref.uri,
                            details={"reason": result.get("skip_reason", "unchanged")},
                        ),
                    )
                else:
                    files_succeeded += 1
                    chunks_created += result.get("chunks_created", 0)
                    all_token_counts.extend(result.get("token_counts", []))

                    # Collect sample output in DRY_RUN mode
                    if sample_outputs is not None and "sample_output" in result:
                        sample_outputs.append(result["sample_output"])

                    await self._emit_progress(
                        progress_callback,
                        ProgressEvent(
                            event_type="file_completed",
                            file_uri=file_ref.uri,
                            details={"chunks_created": result.get("chunks_created", 0)},
                        ),
                    )

                self._failure_tracker.record_success()

            except Exception as e:
                files_failed += 1
                logger.error("Failed to process %s: %s", file_ref.uri, e, exc_info=True)

                # Create failure record
                failure = StageFailure(
                    file_uri=file_ref.uri,
                    stage_id=getattr(e, "stage_id", "unknown"),
                    stage_type=getattr(e, "stage_type", "unknown"),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_traceback=traceback.format_exc(),
                )
                failures.append(failure)

                self._failure_tracker.record_failure(
                    file_ref.uri,
                    failure.stage_id,
                    failure.error_message,
                    error_type=failure.error_type,
                )

                await self._emit_progress(
                    progress_callback,
                    ProgressEvent(
                        event_type="file_failed",
                        file_uri=file_ref.uri,
                        details={"error": str(e), "error_type": failure.error_type},
                    ),
                )

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        # Calculate chunk stats
        chunk_stats = ChunkStats.from_token_counts(all_token_counts)

        # Determine if halted
        halted = self._failure_tracker.should_halt()
        halt_reason = self._failure_tracker.get_halt_reason() if halted else None

        # Emit pipeline completed event
        await self._emit_progress(
            progress_callback,
            ProgressEvent(
                event_type="pipeline_completed",
                details={
                    "files_processed": files_processed,
                    "files_succeeded": files_succeeded,
                    "files_failed": files_failed,
                    "files_skipped": files_skipped,
                    "chunks_created": chunks_created,
                    "halted": halted,
                },
            ),
        )

        return ExecutionResult(
            mode=self.mode,
            files_processed=files_processed,
            files_succeeded=files_succeeded,
            files_failed=files_failed,
            files_skipped=files_skipped,
            chunks_created=chunks_created,
            chunk_stats=chunk_stats,
            failures=failures,
            stage_timings=dict(self._stage_timings),
            total_duration_ms=total_duration_ms,
            sample_outputs=sample_outputs,
            halted=halted,
            halt_reason=halt_reason,
        )

    async def _process_file(
        self,
        file_ref: FileReference,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        """Process a single file through the pipeline.

        Args:
            file_ref: File reference to process
            progress_callback: Optional progress callback

        Returns:
            Dict with processing results
        """
        # 1. Load content
        stage_start = time.perf_counter()
        try:
            load_result = await self._loader.load(file_ref)
        except LoadError as e:
            # Add stage info for error tracking
            e.stage_id = "loader"  # noqa: B010
            e.stage_type = "loader"  # noqa: B010
            raise

        self._record_timing("loader", stage_start)

        # 2. Change detection
        skip_reason = await self._should_skip(file_ref, load_result.content_hash)
        if skip_reason:
            return {"skipped": True, "skip_reason": skip_reason}

        # 3. Route to entry node
        entry_node = self._router.get_entry_node(file_ref)
        if entry_node is None:
            return {"skipped": True, "skip_reason": "no_matching_route"}

        # 4. Execute pipeline stages
        current_node: PipelineNode | None = entry_node
        parsed_text: str = ""
        parse_metadata: dict[str, Any] = {}
        chunks: list[dict[str, Any]] = []
        token_counts: list[int] = []

        while current_node is not None:
            stage_start = time.perf_counter()

            try:
                if current_node.type == NodeType.PARSER:
                    parsed_text, parse_metadata = self._execute_parser(
                        current_node,
                        load_result.content,
                        file_ref,
                    )
                    self._record_timing(f"parser:{current_node.id}", stage_start)

                elif current_node.type == NodeType.CHUNKER:
                    chunks, token_counts = self._execute_chunker(
                        current_node,
                        parsed_text,
                    )
                    self._record_timing(f"chunker:{current_node.id}", stage_start)

                elif current_node.type == NodeType.EXTRACTOR:
                    # Extractor is optional - skip if not implemented
                    self._record_timing(f"extractor:{current_node.id}", stage_start)

                elif current_node.type == NodeType.EMBEDDER:
                    # In FULL mode, we would embed and store here
                    # For now, just record timing
                    self._record_timing(f"embedder:{current_node.id}", stage_start)
                    break  # Embedder is terminal

            except Exception as e:
                # Annotate error with stage info
                e.stage_id = current_node.id  # type: ignore[attr-defined]
                e.stage_type = current_node.type.value  # type: ignore[attr-defined]
                raise

            # Emit stage completed event
            await self._emit_progress(
                progress_callback,
                ProgressEvent(
                    event_type="stage_completed",
                    file_uri=file_ref.uri,
                    stage_id=current_node.id,
                    details={"node_type": current_node.type.value},
                ),
            )

            # Get next node
            next_nodes = self._router.get_next_nodes(current_node, file_ref)
            current_node = next_nodes[0] if next_nodes else None

        # 5. In FULL mode, create document record
        chunks_created = len(chunks)
        if self.mode == ExecutionMode.FULL and chunks_created > 0:
            await self._create_document(file_ref, load_result.content_hash, chunks_created)

        # Prepare result
        result: dict[str, Any] = {
            "skipped": False,
            "chunks_created": chunks_created,
            "token_counts": token_counts,
        }

        # Add sample output in DRY_RUN mode
        if self.mode == ExecutionMode.DRY_RUN:
            result["sample_output"] = SampleOutput(
                file_ref=file_ref,
                chunks=chunks,
                parse_metadata=parse_metadata,
            )

        return result

    async def _should_skip(self, file_ref: FileReference, content_hash: str) -> str | None:
        """Check if file should be skipped due to unchanged content.

        Args:
            file_ref: File reference
            content_hash: SHA-256 hash of content

        Returns:
            Skip reason if file should be skipped, None to process
        """
        if self.mode == ExecutionMode.DRY_RUN:
            # In DRY_RUN mode, don't skip anything
            return None

        try:
            existing = await self._doc_repo.get_by_uri(self.collection_id, file_ref.uri)
            if existing and existing.content_hash == content_hash:
                return "unchanged"
        except Exception as e:
            # Log but don't fail - proceed with processing
            logger.warning("Error checking for existing document: %s", e)

        return None

    def _execute_parser(
        self,
        node: PipelineNode,
        content: bytes,
        file_ref: FileReference,
    ) -> tuple[str, dict[str, Any]]:
        """Execute a parser node.

        Args:
            node: Parser node configuration
            content: Raw file content
            file_ref: File reference with metadata

        Returns:
            Tuple of (parsed text, parse metadata)
        """
        parser = get_parser(node.plugin_id, node.config)

        result = parser.parse_bytes(
            content,
            filename=file_ref.filename,
            file_extension=file_ref.extension,
            mime_type=file_ref.mime_type,
        )

        return result.text, dict(result.metadata)

    def _execute_chunker(
        self,
        node: PipelineNode,
        text: str,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Execute a chunker node.

        Args:
            node: Chunker node configuration
            text: Parsed text to chunk

        Returns:
            Tuple of (list of chunk dicts, list of token counts)
        """
        strategy = UnifiedChunkingFactory.create_strategy(node.plugin_id)

        # Build chunk config from node config
        config = ChunkConfig(
            max_tokens=node.config.get("max_tokens", 1000),
            min_tokens=node.config.get("min_tokens", 100),
            overlap_tokens=node.config.get("overlap_tokens", 50),
            strategy_name=node.plugin_id,
        )

        chunks = strategy.chunk(text, config)

        # Convert to dicts and collect token counts
        chunk_dicts: list[dict[str, Any]] = []
        token_counts: list[int] = []

        for chunk in chunks:
            chunk_dict = {
                "content": chunk.content,
                "metadata": {
                    "chunk_id": chunk.metadata.chunk_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "start_offset": chunk.metadata.start_offset,
                    "end_offset": chunk.metadata.end_offset,
                    "token_count": chunk.metadata.token_count,
                    "hierarchy_level": chunk.metadata.hierarchy_level,
                },
            }
            chunk_dicts.append(chunk_dict)
            token_counts.append(chunk.metadata.token_count)

        return chunk_dicts, token_counts

    async def _create_document(
        self,
        file_ref: FileReference,
        content_hash: str,
        chunk_count: int,  # noqa: ARG002
    ) -> None:
        """Create a document record in the database.

        Args:
            file_ref: File reference
            content_hash: Content SHA-256 hash
            chunk_count: Number of chunks created
        """
        try:
            await self._doc_repo.create(
                collection_id=self.collection_id,
                file_path=file_ref.source_metadata.get("local_path", file_ref.uri),
                file_name=file_ref.filename or file_ref.uri.split("/")[-1],
                file_size=file_ref.size_bytes,
                content_hash=content_hash,
                mime_type=file_ref.mime_type,
                uri=file_ref.uri,
                source_metadata=file_ref.source_metadata,
            )
        except Exception as e:
            logger.error("Failed to create document record: %s", e)
            raise

    def _record_timing(self, stage_key: str, start_time: float) -> None:
        """Record timing for a stage.

        Args:
            stage_key: Unique key for the stage
            start_time: Start time from time.perf_counter()
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if stage_key in self._stage_timings:
            self._stage_timings[stage_key] += elapsed_ms
        else:
            self._stage_timings[stage_key] = elapsed_ms

    async def _emit_progress(
        self,
        callback: ProgressCallback | None,
        event: ProgressEvent,
    ) -> None:
        """Emit a progress event if callback is provided.

        Args:
            callback: Optional async callback
            event: Progress event to emit
        """
        if callback is not None:
            try:
                await callback(event)
            except Exception as e:
                logger.warning("Progress callback failed: %s", e)


__all__ = [
    "PipelineExecutor",
    "PipelineExecutionError",
]
