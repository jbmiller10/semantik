"""Pipeline executor for processing files through the DAG.

This module provides the PipelineExecutor class that orchestrates file
processing through a pipeline DAG, coordinating parsing, chunking,
extraction, and embedding stages.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import httpx

from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.unified.factory import UnifiedChunkingFactory
from shared.database.models import DocumentStatus
from shared.database.repositories.document_repository import DocumentRepository
from shared.pipeline.executor_types import (
    ChunkStats,
    ExecutionMode,
    ExecutionResult,
    PathState,
    ProgressEvent,
    SampleOutput,
    StageFailure,
)
from shared.pipeline.failure_tracker import ConsecutiveFailureTracker
from shared.pipeline.loader import LoadError, PipelineLoader
from shared.pipeline.router import PipelineRouter
from shared.pipeline.sniff import ContentSniffer, SniffCache, SniffConfig, SniffResult
from shared.pipeline.types import FileReference, NodeType, PipelineDAG, PipelineNode
from shared.text_processing.parsers.registry import get_parser

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.connectors.base import BaseConnector


def _get_internal_api_key() -> str:
    """Get the internal API key for VecPipe communication.

    Uses the shared settings which are populated by ensure_internal_api_key()
    at Celery app startup, falling back to environment variable.
    """
    from shared.config import settings

    return settings.INTERNAL_API_KEY or ""


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
            vector_store_name="my_collection",  # Required for FULL mode
            embedding_model="BAAI/bge-small-en-v1.5",
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
        vector_store_name: str | None = None,
        embedding_model: str | None = None,
        quantization: str = "float16",
        sniff_config: SniffConfig | None = None,
    ) -> None:
        """Initialize the pipeline executor.

        Args:
            dag: The pipeline DAG definition
            collection_id: UUID of the collection to populate
            session: Async database session
            connector: Optional connector for loading content from non-file:// URIs
            mode: Execution mode (FULL for production, DRY_RUN for validation)
            consecutive_failure_threshold: Number of consecutive failures before halt
            vector_store_name: Qdrant collection name (required for FULL mode)
            embedding_model: Embedding model name (falls back to DAG config if not provided)
            quantization: Model quantization setting
            sniff_config: Configuration for pre-routing content sniffing

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
        self._vector_store_name = vector_store_name
        self._embedding_model = embedding_model
        self._quantization = quantization

        # Initialize components
        self._router = PipelineRouter(dag)
        self._loader = PipelineLoader(connector)
        # Shared cache for sniff results (survives across files in same execution)
        self._sniff_cache = SniffCache(maxsize=10000, ttl=3600)
        self._sniffer = ContentSniffer(sniff_config, cache=self._sniff_cache)
        self._failure_tracker = ConsecutiveFailureTracker(threshold=consecutive_failure_threshold)

        # Repositories
        self._doc_repo = DocumentRepository(session)

        # Execution state
        self._stage_timings: dict[str, float] = {}
        self._callback_failures: int = 0
        self._sniff_failure_count: int = 0
        self._warnings: list[str] = []

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

        # Reset warnings for this execution run
        self._warnings = []
        # Reset stage timing accumulator for this execution run (aggregated across files)
        self._stage_timings.clear()

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

                    # Collect sample output(s) in DRY_RUN mode
                    if sample_outputs is not None:
                        if "sample_output" in result:
                            sample_outputs.append(result["sample_output"])
                        elif "sample_outputs" in result:
                            sample_outputs.extend(result["sample_outputs"])

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

        # Collect skipped files from connector (if available)
        if self._connector is not None:
            skipped_files = self._connector.get_skipped_files()
            if skipped_files:
                # Summarize to avoid overwhelming warnings for large directories
                if len(skipped_files) <= 5:
                    for path, reason in skipped_files:
                        self._warnings.append(f"Skipped during enumeration: {path}: {reason}")
                else:
                    self._warnings.append(
                        f"Skipped {len(skipped_files)} files during enumeration "
                        f"(e.g., {skipped_files[0][0]}: {skipped_files[0][1]})"
                    )

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
            callback_failures=self._callback_failures,
            warnings=self._warnings if self._warnings else None,
        )

    async def _process_file(
        self,
        file_ref: FileReference,
        progress_callback: ProgressCallback | None,
    ) -> dict[str, Any]:
        """Process a single file through the pipeline.

        Supports parallel fan-out where a document can be processed through
        multiple paths simultaneously (e.g., chunking + summarization).
        Each path produces path-tagged chunks for search filtering.

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
            e.stage_id = "loader"
            e.stage_type = "loader"
            raise

        self._record_timing("loader", stage_start)

        # 2. Pre-routing sniff (enriches file_ref.metadata["detected"])
        # Pass content_hash for cache lookup to speed up reindexing
        stage_start = time.perf_counter()
        try:
            sniff_result = await self._sniffer.sniff(
                load_result.content,
                file_ref,
                content_hash=load_result.content_hash,
            )
        except Exception as e:
            # Sniff failures are non-fatal: we log the error and continue processing.
            # Rationale: It's better to process a document with default routing (via
            # catch-all edges) than to fail the entire document due to a detection
            # issue. The sniff step is an optimization for smarter routing, not a
            # required gate. Errors are recorded in sniff_result.errors for visibility
            # in debugging and monitoring, but don't block the pipeline.
            logger.warning("Sniff failed for %s: %s", file_ref.uri, e, exc_info=True)
            sniff_result = SniffResult(errors=[f"Sniff failed: {e}"])

            # Track sniff failures to surface systemic issues
            self._sniff_failure_count += 1
            if self._sniff_failure_count == 5:
                self._warnings.append("Content detection failing repeatedly. Documents processed with default routing.")
                logger.warning(
                    "Sniff failing repeatedly (%d times). This may indicate a systemic issue.",
                    self._sniff_failure_count,
                )

        # Always enrich with whatever results we have (may include errors)
        self._sniffer.enrich_file_ref(file_ref, sniff_result)

        # Collect sniff errors as warnings (non-fatal but user-visible)
        if sniff_result.errors:
            for err in sniff_result.errors:
                self._warnings.append(f"{file_ref.uri}: {err}")
        self._record_timing("sniff", stage_start)

        # 3. Change detection
        skip_reason = await self._should_skip(file_ref, load_result.content_hash)
        if skip_reason:
            return {"skipped": True, "skip_reason": skip_reason}

        # 4. Route to entry nodes (may be multiple for parallel fan-out)
        entry_nodes = self._router.get_entry_nodes(file_ref)
        if not entry_nodes:
            return {"skipped": True, "skip_reason": "no_matching_route"}

        # 5. Initialize path states for each entry point using a deque for work queue pattern
        # Using deque allows mid-pipeline fan-out branches to be added and executed
        path_queue: deque[PathState] = deque()
        completed_paths: list[PathState] = []
        for node, path_name in entry_nodes:
            path_queue.append(PathState(path_id=path_name, current_node=node))

        # 6. Create single document record (shared across all paths)
        doc_id: str | None = None
        if self.mode == ExecutionMode.FULL:
            doc_id = await self._create_document(file_ref, load_result.content_hash)

        # 7. Execute each path sequentially (avoids GPU OOM; Celery parallelizes across docs)
        # The deque allows branched paths to be added and executed during iteration
        all_chunks: list[dict[str, Any]] = []
        all_token_counts: list[int] = []
        all_sample_outputs: list[SampleOutput] = []
        first_path_error: Exception | None = None

        while path_queue:
            path_state = path_queue.popleft()
            try:
                await self._execute_path(
                    path_state=path_state,
                    path_queue=path_queue,
                    file_ref=file_ref,
                    load_content=load_result.content,
                    doc_id=doc_id,
                    progress_callback=progress_callback,
                )
                # Collect results from this path
                all_chunks.extend(path_state.chunks)
                all_token_counts.extend(path_state.token_counts)

                # Collect sample output in DRY_RUN mode
                if self.mode == ExecutionMode.DRY_RUN:
                    all_sample_outputs.append(
                        SampleOutput(
                            file_ref=file_ref,
                            chunks=path_state.chunks,
                            parse_metadata=path_state.parse_metadata,
                            path_id=path_state.path_id,
                        )
                    )
            except Exception as e:
                # Mark path as failed but continue with other paths
                if first_path_error is None:
                    first_path_error = e
                path_state.error = e
                path_state.completed = True
                logger.error(
                    "Path %s failed for %s: %s",
                    path_state.path_id,
                    file_ref.uri,
                    e,
                    exc_info=True,
                )

            # Track completed paths (both successful and failed)
            completed_paths.append(path_state)

        # 8. Update document status if we created chunks
        # If all paths failed, raise after every path had a chance to run.
        if first_path_error is not None and all(ps.error is not None for ps in completed_paths):
            if self.mode == ExecutionMode.FULL and doc_id:
                error_msg = str(first_path_error)[:500]
                await self._doc_repo.update_status(
                    doc_id,
                    DocumentStatus.FAILED,
                    error_message=error_msg,
                )
                await self.session.commit()
            raise first_path_error

        if self.mode == ExecutionMode.FULL and doc_id:
            failed_paths = [ps for ps in completed_paths if ps.error is not None]

            if all_chunks:
                # We have chunks - mark status based on whether any paths failed
                if failed_paths:
                    # Include exception type and message for debugging
                    error_details = [f"{ps.path_id}: {type(ps.error).__name__}: {ps.error}" for ps in failed_paths]
                    error_msg = "Paths failed: " + "; ".join(error_details)
                    await self._doc_repo.update_status(
                        doc_id,
                        DocumentStatus.FAILED,
                        error_message=error_msg[:500],
                    )
                else:
                    await self._doc_repo.update_status(
                        doc_id,
                        DocumentStatus.COMPLETED,
                        chunk_count=len(all_chunks),
                    )
            elif failed_paths:
                # No chunks but some paths failed - mark as failed
                # Include exception type and message for debugging
                error_details = [f"{ps.path_id}: {type(ps.error).__name__}: {ps.error}" for ps in failed_paths]
                error_msg = "Paths failed: " + "; ".join(error_details)
                await self._doc_repo.update_status(
                    doc_id,
                    DocumentStatus.FAILED,
                    error_message=error_msg[:500],
                )
            else:
                # No chunks and no failures - document produced no output (e.g., empty file)
                # Mark as completed with 0 chunks
                await self._doc_repo.update_status(
                    doc_id,
                    DocumentStatus.COMPLETED,
                    chunk_count=0,
                )
            await self.session.commit()

        # 9. Prepare result
        # Collect failed path information for visibility
        failed_path_info = [
            {"path_id": ps.path_id, "error": str(ps.error)} for ps in completed_paths if ps.error is not None
        ]

        result: dict[str, Any] = {
            "skipped": False,
            "chunks_created": len(all_chunks),
            "token_counts": all_token_counts,
            "failed_paths": failed_path_info,  # Empty list if all succeeded
        }

        # Add sample output in DRY_RUN mode
        if self.mode == ExecutionMode.DRY_RUN:
            if len(all_sample_outputs) == 1:
                result["sample_output"] = all_sample_outputs[0]
            else:
                result["sample_outputs"] = all_sample_outputs

        return result

    async def _execute_path(
        self,
        path_state: PathState,
        path_queue: deque[PathState],
        file_ref: FileReference,
        load_content: bytes,
        doc_id: str | None,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Execute a single path through the pipeline DAG.

        Processes stages sequentially: parser -> chunker -> extractor -> embedder.
        Results are accumulated in path_state.

        Args:
            path_state: State object for this execution path
            path_queue: Work queue for path states (used to append branched paths for execution)
            file_ref: File reference being processed
            load_content: Raw file content bytes
            doc_id: Document ID for Qdrant payloads (None in DRY_RUN mode)
            progress_callback: Optional progress callback
        """
        current_node = path_state.current_node

        while current_node is not None:
            path_state.current_node = current_node
            stage_start = time.perf_counter()

            try:
                if current_node.type == NodeType.PARSER:
                    path_state.parsed_text, path_state.parse_metadata = self._execute_parser(
                        current_node,
                        load_content,
                        file_ref,
                    )
                    self._record_timing(f"parser:{current_node.id}:{path_state.path_id}", stage_start)

                    # Enrich file_ref with parsed metadata for mid-pipeline routing
                    self._enrich_parsed_metadata(file_ref, path_state.parse_metadata)

                elif current_node.type == NodeType.CHUNKER:
                    chunks, token_counts = self._execute_chunker(
                        current_node,
                        path_state.parsed_text,
                    )
                    # Tag chunks with path_id
                    for chunk in chunks:
                        chunk["path_id"] = path_state.path_id
                    path_state.chunks = chunks
                    path_state.token_counts = token_counts
                    self._record_timing(f"chunker:{current_node.id}:{path_state.path_id}", stage_start)

                elif current_node.type == NodeType.EXTRACTOR:
                    # Extractor is optional - skip if not implemented
                    self._record_timing(f"extractor:{current_node.id}:{path_state.path_id}", stage_start)

                elif current_node.type == NodeType.EMBEDDER:
                    # Embedder stage - embed chunks and store in Qdrant
                    if self.mode == ExecutionMode.FULL and path_state.chunks and doc_id:
                        try:
                            await self._execute_embedder(
                                node=current_node,
                                chunks=path_state.chunks,
                                file_ref=file_ref,
                                doc_id=doc_id,
                                path_id=path_state.path_id,
                            )
                        except Exception as embed_error:
                            # Mark path as failed
                            path_state.error = embed_error
                            raise

                    self._record_timing(f"embedder:{current_node.id}:{path_state.path_id}", stage_start)
                    path_state.completed = True
                    break  # Embedder is terminal

            except Exception as e:
                # Annotate error with stage info
                e.stage_id = current_node.id  # type: ignore[attr-defined]
                e.stage_type = current_node.type.value  # type: ignore[attr-defined]
                e.path_id = path_state.path_id  # type: ignore[attr-defined]
                raise

            # Emit stage completed event
            await self._emit_progress(
                progress_callback,
                ProgressEvent(
                    event_type="stage_completed",
                    file_uri=file_ref.uri,
                    stage_id=current_node.id,
                    details={
                        "node_type": current_node.type.value,
                        "path_id": path_state.path_id,
                    },
                ),
            )

            # Get next node for this path
            routed_next = self._router.get_next_nodes_with_paths(current_node, file_ref)
            if not routed_next:
                current_node = None
                break

            # Mid-pipeline fan-out algorithm:
            # When routing returns multiple edges (parallel branches), we must:
            # 1. Keep processing the first branch in-place (this path_state)
            # 2. Queue new PathState copies for additional branches
            # 3. Deep copy chunks since they may be modified by downstream stages
            # 4. Re-tag chunks with new path_id for search filtering

            # No fan-out: continue on the same path_id for backward compatibility.
            if len(routed_next) == 1:
                current_node = routed_next[0][0]
                continue

            # Fan-out: create additional path states for remaining branches, and
            # update the current path to follow the first branch.
            for next_node, next_path_id in routed_next[1:]:
                branched_state = PathState(
                    path_id=next_path_id,
                    current_node=next_node,
                    parsed_text=path_state.parsed_text,
                    parse_metadata=dict(path_state.parse_metadata),
                    chunks=deepcopy(path_state.chunks),
                    token_counts=list(path_state.token_counts),
                )
                if branched_state.chunks:
                    for chunk in branched_state.chunks:
                        chunk["path_id"] = next_path_id
                path_queue.append(branched_state)

            # Follow the first branch in this path state
            first_next_node, first_next_path_id = routed_next[0]
            if path_state.chunks and path_state.path_id != first_next_path_id:
                for chunk in path_state.chunks:
                    chunk["path_id"] = first_next_path_id
            path_state.path_id = first_next_path_id
            current_node = first_next_node

        path_state.completed = True

    async def _should_skip(self, file_ref: FileReference, content_hash: str) -> str | None:
        """Check if file should be skipped due to unchanged content.

        Args:
            file_ref: File reference
            content_hash: SHA-256 hash of content

        Returns:
            Skip reason if file should be skipped, None to process

        Raises:
            Exception: Database errors propagate to caller for proper handling
        """
        if self.mode == ExecutionMode.DRY_RUN:
            # In DRY_RUN mode, don't skip anything
            return None

        # get_by_uri returns None if document doesn't exist
        # DB connectivity/transaction errors are annotated with stage info
        try:
            existing = await self._doc_repo.get_by_uri(self.collection_id, file_ref.uri)
        except Exception as e:
            # Add stage context for error tracking
            e.stage_id = "skip_check"  # type: ignore[attr-defined]
            e.stage_type = "database"  # type: ignore[attr-defined]
            raise

        if existing is not None and existing.content_hash == content_hash:
            return "unchanged"

        return None

    def _enrich_parsed_metadata(
        self,
        file_ref: FileReference,
        parse_metadata: dict[str, Any],
    ) -> None:
        """Enrich FileReference with parsed metadata for mid-pipeline routing.

        Copies standardized parsed.* fields from parser output to
        file_ref.metadata["parsed"] for use in routing predicates.

        Args:
            file_ref: File reference to enrich
            parse_metadata: Metadata dict from parser output
        """
        if not parse_metadata:
            return

        # Recognized parsed.* field names from ParsedMetadata schema
        parsed_fields = {
            "page_count",
            "has_tables",
            "has_images",
            "has_code_blocks",
            "detected_language",
            "approx_token_count",
            "line_count",
            "element_types",
            "text_quality",
        }

        # Initialize parsed namespace if needed
        if "parsed" not in file_ref.metadata:
            file_ref.metadata["parsed"] = {}

        # Copy only recognized fields
        for key, value in parse_metadata.items():
            if key in parsed_fields:
                file_ref.metadata["parsed"][key] = value

    def _execute_parser(
        self,
        node: PipelineNode,
        content: bytes,
        file_ref: FileReference,
    ) -> tuple[str, dict[str, Any]]:
        """Execute a parser node.

        Uses the plugin registry first for parser lookup, falling back to
        the legacy registry for backward compatibility during migration.

        Args:
            node: Parser node configuration
            content: Raw file content
            file_ref: File reference with metadata

        Returns:
            Tuple of (parsed text, parse metadata)
        """
        # Try plugin registry first (new plugin system)
        from shared.plugins import plugin_registry

        record = plugin_registry.get("parser", node.plugin_id)
        if record is not None:
            parser_cls = record.plugin_class
            parser = parser_cls(node.config)
            result = parser.parse_bytes(
                content,
                filename=file_ref.filename,
                file_extension=file_ref.extension,
                mime_type=file_ref.mime_type,
            )
            return result.text, dict(result.metadata)

        # Fallback to legacy registry (deprecation path)
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
                "chunk_id": chunk.metadata.chunk_id,  # Top-level for build_chunk_point compatibility
                "content": chunk.content,
                "metadata": {
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

    async def _execute_embedder(
        self,
        node: PipelineNode,
        chunks: list[dict[str, Any]],
        file_ref: FileReference,
        doc_id: str,
        path_id: str = "default",
    ) -> None:
        """Execute embedder node - embed chunks and store in Qdrant.

        Args:
            node: Embedder node configuration
            chunks: List of chunk dicts from chunker
            file_ref: Original file reference
            doc_id: Document ID for the chunk payloads
            path_id: Path identifier for parallel fan-out (default: "default")
        """
        if not self._vector_store_name:
            raise ValueError("vector_store_name required for embedding in FULL mode")

        # Get embedding model from node config or instance default
        embedding_model = node.config.get("model") or self._embedding_model
        if not embedding_model:
            raise ValueError("Embedding model not configured")

        # Extract texts from chunks
        texts = [chunk.get("content") or chunk.get("text") or "" for chunk in chunks]
        if not texts:
            logger.warning("No texts to embed for %s", file_ref.uri)
            return

        # Build headers for internal API calls
        headers = {
            "X-Internal-Api-Key": _get_internal_api_key(),
            "Content-Type": "application/json",
        }

        # Call VecPipe /embed endpoint
        from shared.config import settings

        vecpipe_url = f"{settings.SEARCH_API_URL}/embed"
        embed_request = {
            "texts": texts,
            "model_name": embedding_model,
            "quantization": self._quantization,
            "mode": "document",  # Document indexing uses document mode
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            logger.info("Calling vecpipe /embed for %d texts", len(texts))
            response = await client.post(vecpipe_url, json=embed_request, headers=headers)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to generate embeddings via vecpipe: {response.status_code} - {response.text}"
                )

            embed_response = response.json()
            embeddings = embed_response.get("embeddings")

        if embeddings is None:
            raise RuntimeError("VecPipe returned no embeddings field in response")
        if len(embeddings) == 0:
            raise RuntimeError(f"VecPipe returned empty embeddings for {len(texts)} texts")

        if len(embeddings) != len(chunks):
            raise RuntimeError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunks)}")

        # Build points for Qdrant
        path = file_ref.metadata.get("source", {}).get("local_path", file_ref.uri)
        total_chunks = len(chunks)
        points = []

        for i, chunk in enumerate(chunks):
            # chunk_id is at top level from _execute_chunker
            chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
            # path_id from chunk takes precedence (set by _execute_path), fall back to parameter
            chunk_path_id = chunk.get("path_id", path_id)
            point = {
                "id": str(uuid.uuid4()),
                "vector": embeddings[i],
                "payload": {
                    "collection_id": self.collection_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "path": path,
                    "content": chunk.get("content") or chunk.get("text") or "",
                    "metadata": chunk.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "path_id": chunk_path_id,
                },
            }
            points.append(point)

        # Upsert to Qdrant via VecPipe in batches
        batch_size = 100
        vecpipe_upsert_url = f"{settings.SEARCH_API_URL}/upsert"

        async with httpx.AsyncClient(timeout=60.0) as client:
            for batch_start in range(0, len(points), batch_size):
                batch_end = min(batch_start + batch_size, len(points))
                batch_points = points[batch_start:batch_end]

                upsert_request = {
                    "collection_name": self._vector_store_name,
                    "points": batch_points,
                    "wait": True,
                }

                response = await client.post(vecpipe_upsert_url, json=upsert_request, headers=headers)

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to upsert vectors via vecpipe: {response.status_code} - {response.text}"
                    )

        # Note: Document status update is handled by _process_file after all paths complete.
        # This ensures proper status accounting when multiple paths fan out from a single document.

        logger.info(
            "Embedded and stored %d chunks for %s in %s",
            len(chunks),
            file_ref.uri,
            self._vector_store_name,
        )

    async def _create_document(
        self,
        file_ref: FileReference,
        content_hash: str,
    ) -> str:
        """Create a document record in the database.

        Args:
            file_ref: File reference
            content_hash: Content SHA-256 hash

        Returns:
            The document ID (UUID string)
        """
        try:
            doc = await self._doc_repo.create(
                collection_id=self.collection_id,
                file_path=file_ref.metadata.get("source", {}).get("local_path", file_ref.uri),
                file_name=file_ref.filename or file_ref.uri.split("/")[-1],
                file_size=file_ref.size_bytes,
                content_hash=content_hash,
                mime_type=file_ref.mime_type,
                uri=file_ref.uri,
                # DB column is still called source_metadata - pass the source namespace
                source_metadata=file_ref.metadata.get("source", {}),
            )
            await self.session.commit()
            return str(doc.id)
        except Exception as e:
            logger.error("Failed to create document record for %s: %s", file_ref.uri, e)
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
                self._callback_failures += 1
                logger.warning(
                    "Progress callback failed for %s (failure #%d): %s",
                    event.event_type,
                    self._callback_failures,
                    e,
                    exc_info=True,
                )
                if self._callback_failures == 5:
                    logger.error("Progress callbacks failing repeatedly (%d failures)", self._callback_failures)
                    self._warnings.append("Progress reporting degraded: real-time updates may be unavailable")


__all__ = [
    "PipelineExecutor",
    "PipelineExecutionError",
]
