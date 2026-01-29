"""Pipeline route preview service.

This service handles previewing how a file would be routed through a pipeline DAG
without actually processing the file. It provides detailed information about
predicate evaluation at each routing stage.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.pipeline.predicates import get_nested_value, match_value
from shared.pipeline.router import PipelineRouter
from shared.pipeline.sniff import ContentSniffer, SniffConfig
from shared.pipeline.types import FileReference, NodeType, PipelineDAG
from shared.pipeline.validation import SOURCE_NODE
from shared.text_processing.parsers.registry import get_parser
from webui.api.v2.pipeline_schemas import (
    EdgeEvaluationResult,
    FieldEvaluationResult,
    PathInfo,
    RoutePreviewResponse,
    StageEvaluationResult,
)

if TYPE_CHECKING:
    from shared.pipeline.types import PipelineEdge, PipelineNode

logger = logging.getLogger(__name__)


class PipelinePreviewService:
    """Service for previewing pipeline routing decisions.

    This service evaluates how a file would be routed through a pipeline DAG
    and provides detailed information about predicate evaluation at each stage.

    Example:
        >>> service = PipelinePreviewService()
        >>> result = await service.preview_route(
        ...     file_content=pdf_bytes,
        ...     filename="document.pdf",
        ...     dag=my_dag,
        ... )
        >>> print(result.path)
        ['_source', 'pdf_parser', 'recursive_chunker', 'embedder']
    """

    def __init__(self, sniff_config: SniffConfig | None = None) -> None:
        """Initialize the preview service.

        Args:
            sniff_config: Configuration for content sniffing
        """
        self._sniffer = ContentSniffer(sniff_config or SniffConfig())

    async def preview_route(
        self,
        file_content: bytes,
        filename: str,
        dag: dict[str, Any],
        include_parser_metadata: bool = True,
    ) -> RoutePreviewResponse:
        """Preview how a file would be routed through the pipeline.

        Supports parallel fan-out at entry routing. When multiple entry edges
        match (due to parallel=True), all resulting paths are computed and
        returned in the `paths` field.

        Args:
            file_content: Raw file content bytes
            filename: Original filename
            dag: Pipeline DAG definition as dictionary
            include_parser_metadata: Whether to run the parser for metadata

        Returns:
            RoutePreviewResponse with detailed routing information
        """
        start_time = time.perf_counter()
        warnings: list[str] = []

        # 1. Build FileReference
        file_ref = self._build_file_reference(file_content, filename)

        # 2. Parse and validate DAG
        try:
            dag_obj = PipelineDAG.from_dict(dag)
            errors = dag_obj.validate()
            if errors:
                warnings.extend(f"DAG validation: {e.message}" for e in errors)
        except Exception as e:
            raise ValueError(f"Invalid DAG: {e}") from e

        # 3. Run content sniffer
        sniff_result = await self._sniffer.sniff(file_content, file_ref)
        self._sniffer.enrich_file_ref(file_ref, sniff_result)

        # Build sniff result dict
        sniff_dict = sniff_result.to_metadata_dict()
        if sniff_result.errors:
            warnings.extend(sniff_result.errors)

        # 4. Create router and evaluate routing
        router = PipelineRouter(dag_obj)
        routing_stages: list[StageEvaluationResult] = []

        # Evaluate entry routing (_source -> first node(s))
        entry_stage = self._evaluate_entry_routing(dag_obj, router, file_ref)
        routing_stages.append(entry_stage)

        if entry_stage.selected_node is None:
            # No route found
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            return RoutePreviewResponse(
                file_info=self._build_file_info(file_ref),
                sniff_result=sniff_dict if sniff_dict else None,
                routing_stages=routing_stages,
                path=[SOURCE_NODE],
                paths=None,
                parsed_metadata=None,
                total_duration_ms=total_duration_ms,
                warnings=warnings + ["No matching route found from source"],
            )

        # Get all entry nodes (for parallel fan-out)
        entry_nodes = entry_stage.selected_nodes or [entry_stage.selected_node]

        # Build path_name mapping from evaluated edges
        path_names: dict[str, str] = {}
        for edge_result in entry_stage.evaluated_edges:
            if edge_result.status in ("matched", "matched_parallel"):
                path_names[edge_result.to_node] = edge_result.path_name or edge_result.to_node

        # 5. Walk through the pipeline for each entry node
        all_paths: list[PathInfo] = []
        primary_path: list[str] = [SOURCE_NODE]
        parsed_metadata: dict[str, Any] | None = None
        ran_parser_for_primary = False

        for idx, entry_node_id in enumerate(entry_nodes):
            entry_path_name = path_names.get(entry_node_id, entry_node_id)

            # Stack of (current_path, current_node_id, current_path_name, is_primary_path)
            stack: list[tuple[list[str], str, str, bool]] = [
                ([SOURCE_NODE, entry_node_id], entry_node_id, entry_path_name, idx == 0)
            ]

            while stack:
                current_path, current_node_id, current_path_name, is_primary_path = stack.pop()
                current_node = router.get_node(current_node_id)

                while current_node is not None:
                    # Handle parser stage (only for the primary path to avoid duplicate work)
                    if (
                        is_primary_path
                        and not ran_parser_for_primary
                        and current_node.type == NodeType.PARSER
                        and include_parser_metadata
                    ):
                        ran_parser_for_primary = True
                        try:
                            parsed_metadata = self._run_parser(current_node, file_content, file_ref)
                            # Enrich file_ref with parsed metadata for mid-pipeline routing
                            self._enrich_parsed_metadata(file_ref, parsed_metadata)
                        except Exception as e:
                            parser_id = current_node.plugin_id
                            fname = file_ref.filename or file_ref.uri
                            error_type = type(e).__name__

                            warnings.append(f"Parser '{parser_id}' failed on '{fname}': {error_type}: {e}")
                            logger.warning(
                                "Parser %s failed during preview for %s: %s",
                                parser_id,
                                fname,
                                e,
                                exc_info=True,
                            )

                    # Evaluate next routing stage (only record for primary path to avoid duplicates)
                    next_stage = self._evaluate_next_routing(dag_obj, router, current_node, file_ref)
                    if next_stage is None:
                        break

                    if is_primary_path:
                        routing_stages.append(next_stage)

                    selected_edges = [
                        e for e in next_stage.evaluated_edges if e.status in ("matched", "matched_parallel")
                    ]
                    if not selected_edges:
                        break

                    # Fan-out: branch to all selected nodes, keep walking the primary branch.
                    if len(selected_edges) > 1:
                        for edge_result in selected_edges[1:]:
                            branched_path = current_path + [edge_result.to_node]
                            branched_name = edge_result.path_name or edge_result.to_node
                            stack.append((branched_path, edge_result.to_node, branched_name, False))

                        first_edge = selected_edges[0]
                        current_path.append(first_edge.to_node)
                        current_path_name = first_edge.path_name or first_edge.to_node
                        current_node = router.get_node(first_edge.to_node)
                        continue

                    # Single next node
                    edge_result = selected_edges[0]
                    current_path.append(edge_result.to_node)
                    if edge_result.path_name is not None:
                        current_path_name = edge_result.path_name
                    current_node = router.get_node(edge_result.to_node)

                all_paths.append(PathInfo(path_name=current_path_name, nodes=current_path))
                if is_primary_path:
                    primary_path = current_path

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        return RoutePreviewResponse(
            file_info=self._build_file_info(file_ref),
            sniff_result=sniff_dict if sniff_dict else None,
            routing_stages=routing_stages,
            path=primary_path,
            paths=all_paths if len(all_paths) > 1 else None,
            parsed_metadata=parsed_metadata,
            total_duration_ms=total_duration_ms,
            warnings=warnings,
        )

    def _build_file_reference(self, content: bytes, filename: str) -> FileReference:
        """Build a FileReference from uploaded content.

        Args:
            content: File content bytes
            filename: Original filename

        Returns:
            FileReference with metadata
        """
        # Sanitize filename to prevent path traversal attacks
        # Path.name extracts only the basename, stripping any directory components
        safe_filename = Path(filename).name

        # Extract extension from sanitized filename
        path = Path(safe_filename)
        extension = path.suffix.lower() if path.suffix else None

        # Guess MIME type from sanitized filename
        mime_type, _ = mimetypes.guess_type(safe_filename)

        # Compute content hash
        content_hash = hashlib.sha256(content).hexdigest()

        return FileReference(
            uri=f"preview://{safe_filename}",
            source_type="preview",
            content_type="document",
            filename=safe_filename,
            extension=extension,
            mime_type=mime_type,
            size_bytes=len(content),
            change_hint=content_hash,
            metadata={"source": {"filename": safe_filename, "preview": True}},
        )

    def _build_file_info(self, file_ref: FileReference) -> dict[str, Any]:
        """Build file info dict from FileReference.

        Args:
            file_ref: File reference

        Returns:
            Dict with file information
        """
        return {
            "filename": file_ref.filename,
            "extension": file_ref.extension,
            "mime_type": file_ref.mime_type,
            "size_bytes": file_ref.size_bytes,
            "uri": file_ref.uri,
        }

    def _evaluate_entry_routing(
        self,
        dag: PipelineDAG,
        router: PipelineRouter,  # noqa: ARG002 - matches _evaluate_next_routing signature
        file_ref: FileReference,
    ) -> StageEvaluationResult:
        """Evaluate routing from _source to entry node using 4-step evaluation order.

        The evaluation follows the router's actual logic:
        1. Parallel predicate edges - all matching edges fire
        2. Exclusive predicate edges - first match wins
        3. Parallel catch-all edges - all fire if no exclusive matched
        4. Exclusive catch-all edges - first wins if no exclusive matched

        Args:
            dag: Pipeline DAG
            router: Pipeline router
            file_ref: File reference

        Returns:
            StageEvaluationResult with edge evaluations and selected nodes
        """
        # Get all edges from _source
        source_edges = [e for e in dag.edges if e.from_node == SOURCE_NODE]

        # Categorize edges into 4 groups
        parallel_predicate: list[PipelineEdge] = []
        exclusive_predicate: list[PipelineEdge] = []
        parallel_catchall: list[PipelineEdge] = []
        exclusive_catchall: list[PipelineEdge] = []

        for edge in source_edges:
            is_catchall = edge.when is None or edge.when == {}
            if edge.parallel:
                if is_catchall:
                    parallel_catchall.append(edge)
                else:
                    parallel_predicate.append(edge)
            else:
                if is_catchall:
                    exclusive_catchall.append(edge)
                else:
                    exclusive_predicate.append(edge)

        evaluated_edges: list[EdgeEvaluationResult] = []
        selected_nodes: list[str] = []
        path_names: dict[str, str] = {}  # node_id -> path_name
        exclusive_matched = False

        # Step 1: Evaluate parallel predicate edges (all matching ones fire)
        for edge in parallel_predicate:
            field_evals = self._evaluate_predicate_fields(file_ref, edge.when)
            matched = all(fe.matched for fe in field_evals)

            if matched:
                selected_nodes.append(edge.to_node)
                path_names[edge.to_node] = edge.path_name or edge.to_node
                status = "matched_parallel"
            else:
                status = "not_matched"

            evaluated_edges.append(
                EdgeEvaluationResult(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    predicate=edge.when,
                    matched=matched,
                    status=status,
                    field_evaluations=field_evals,
                    is_parallel=True,
                    path_name=edge.path_name,
                )
            )

        # Step 2: Evaluate exclusive predicate edges (first match wins)
        for edge in exclusive_predicate:
            if exclusive_matched:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=edge.when,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                        is_parallel=False,
                        path_name=edge.path_name,
                    )
                )
            else:
                field_evals = self._evaluate_predicate_fields(file_ref, edge.when)
                matched = all(fe.matched for fe in field_evals)

                if matched:
                    selected_nodes.append(edge.to_node)
                    path_names[edge.to_node] = edge.path_name or edge.to_node
                    exclusive_matched = True
                    status = "matched"
                else:
                    status = "not_matched"

                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=edge.when,
                        matched=matched,
                        status=status,
                        field_evaluations=field_evals,
                        is_parallel=False,
                        path_name=edge.path_name,
                    )
                )

        # Step 3: Evaluate parallel catch-all edges (all fire if no exclusive matched)
        for edge in parallel_catchall:
            # Parallel catch-all always matches (and fires), even if an exclusive
            # predicate matched. This mirrors PipelineRouter.get_entry_nodes().
            selected_nodes.append(edge.to_node)
            path_names[edge.to_node] = edge.path_name or edge.to_node
            evaluated_edges.append(
                EdgeEvaluationResult(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    predicate=None,
                    matched=True,
                    status="matched_parallel",
                    field_evaluations=None,
                    is_parallel=True,
                    path_name=edge.path_name,
                )
            )

        # Step 4: Evaluate exclusive catch-all edges (first wins if no exclusive matched)
        for edge in exclusive_catchall:
            if exclusive_matched:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                        is_parallel=False,
                        path_name=edge.path_name,
                    )
                )
            else:
                # Exclusive catch-all: first one wins
                selected_nodes.append(edge.to_node)
                path_names[edge.to_node] = edge.path_name or edge.to_node
                exclusive_matched = True
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=True,
                        status="matched",
                        field_evaluations=None,
                        is_parallel=False,
                        path_name=edge.path_name,
                    )
                )

        # Determine primary selected node (first in list)
        selected_node = selected_nodes[0] if selected_nodes else None

        return StageEvaluationResult(
            stage="entry_routing",
            from_node=SOURCE_NODE,
            evaluated_edges=evaluated_edges,
            selected_node=selected_node,
            selected_nodes=selected_nodes if len(selected_nodes) > 1 else None,
            metadata_snapshot=dict(file_ref.metadata),
        )

    def _evaluate_next_routing(
        self,
        dag: PipelineDAG,
        router: PipelineRouter,  # noqa: ARG002 - matches _evaluate_entry_routing signature
        current_node: PipelineNode,
        file_ref: FileReference,
    ) -> StageEvaluationResult | None:
        """Evaluate routing from current node to next node.

        Args:
            dag: Pipeline DAG
            router: Pipeline router
            current_node: Current node
            file_ref: File reference

        Returns:
            StageEvaluationResult or None if no outgoing edges
        """
        # Get outgoing edges from current node
        outgoing_edges = [e for e in dag.edges if e.from_node == current_node.id]

        if not outgoing_edges:
            return None

        # Categorize edges by parallel flag and predicate presence
        parallel_predicate: list[PipelineEdge] = []
        parallel_catchall: list[PipelineEdge] = []
        exclusive_predicate: list[PipelineEdge] = []
        exclusive_catchall: list[PipelineEdge] = []

        for edge in outgoing_edges:
            is_catchall = edge.when is None or edge.when == {}
            if edge.parallel:
                if is_catchall:
                    parallel_catchall.append(edge)
                else:
                    parallel_predicate.append(edge)
            else:
                if is_catchall:
                    exclusive_catchall.append(edge)
                else:
                    exclusive_predicate.append(edge)

        evaluated_edges: list[EdgeEvaluationResult] = []
        selected_nodes: list[str] = []
        exclusive_matched = False

        # Step 1: Evaluate parallel predicate edges (all matching ones fire)
        for edge in parallel_predicate:
            field_evals = self._evaluate_predicate_fields(file_ref, edge.when)
            matched = all(fe.matched for fe in field_evals)

            if matched:
                selected_nodes.append(edge.to_node)
                status = "matched_parallel"
            else:
                status = "not_matched"

            evaluated_edges.append(
                EdgeEvaluationResult(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    predicate=edge.when,
                    matched=matched,
                    status=status,
                    field_evaluations=field_evals,
                    is_parallel=True,
                    path_name=edge.path_name,
                )
            )

        # Step 2: Evaluate exclusive predicate edges (first match wins)
        for edge in exclusive_predicate:
            if exclusive_matched:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=edge.when,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                        is_parallel=False,
                        path_name=edge.path_name,
                    )
                )
                continue

            field_evals = self._evaluate_predicate_fields(file_ref, edge.when)
            matched = all(fe.matched for fe in field_evals)

            if matched:
                selected_nodes.append(edge.to_node)
                exclusive_matched = True
                status = "matched"
            else:
                status = "not_matched"

            evaluated_edges.append(
                EdgeEvaluationResult(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    predicate=edge.when,
                    matched=matched,
                    status=status,
                    field_evaluations=field_evals,
                    is_parallel=False,
                    path_name=edge.path_name,
                )
            )

        # Step 3: Evaluate parallel catch-all edges (all fire)
        for edge in parallel_catchall:
            selected_nodes.append(edge.to_node)
            evaluated_edges.append(
                EdgeEvaluationResult(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    predicate=None,
                    matched=True,
                    status="matched_parallel",
                    field_evaluations=None,
                    is_parallel=True,
                    path_name=edge.path_name,
                )
            )

        # Step 4: Evaluate exclusive catch-all edges (first wins if no exclusive predicate matched)
        for edge in exclusive_catchall:
            if exclusive_matched:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                        is_parallel=False,
                        path_name=edge.path_name,
                    )
                )
                continue

            selected_nodes.append(edge.to_node)
            exclusive_matched = True
            evaluated_edges.append(
                EdgeEvaluationResult(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    predicate=None,
                    matched=True,
                    status="matched",
                    field_evaluations=None,
                    is_parallel=False,
                    path_name=edge.path_name,
                )
            )

        stage_name = f"{current_node.type.value}_to_next"
        selected_node = selected_nodes[0] if selected_nodes else None
        return StageEvaluationResult(
            stage=stage_name,
            from_node=current_node.id,
            evaluated_edges=evaluated_edges,
            selected_node=selected_node,
            selected_nodes=selected_nodes if len(selected_nodes) > 1 else None,
            metadata_snapshot=dict(file_ref.metadata),
        )

    def _evaluate_predicate_fields(
        self,
        file_ref: FileReference,
        predicate: dict[str, Any] | None,
    ) -> list[FieldEvaluationResult]:
        """Evaluate each field in a predicate.

        Args:
            file_ref: File reference
            predicate: Predicate dict

        Returns:
            List of field evaluation results
        """
        if predicate is None or not predicate:
            return []

        results: list[FieldEvaluationResult] = []

        for field, pattern in predicate.items():
            # Translate legacy paths
            translated_field = field
            if field.startswith("source_metadata."):
                translated_field = field.replace("source_metadata.", "metadata.source.", 1)
            elif field == "source_metadata":
                translated_field = "metadata.source"

            value = get_nested_value(file_ref, translated_field)
            matched = match_value(pattern, value)

            results.append(
                FieldEvaluationResult(
                    field=field,
                    pattern=pattern,
                    value=value,
                    matched=matched,
                )
            )

        return results

    def _run_parser(
        self,
        node: PipelineNode,
        content: bytes,
        file_ref: FileReference,
    ) -> dict[str, Any]:
        """Run a parser node to extract metadata.

        Args:
            node: Parser node
            content: File content
            file_ref: File reference

        Returns:
            Parser metadata dict
        """
        # Try plugin registry first
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
            return dict(result.metadata)

        # Fallback to legacy registry
        parser = get_parser(node.plugin_id, node.config)
        result = parser.parse_bytes(
            content,
            filename=file_ref.filename,
            file_extension=file_ref.extension,
            mime_type=file_ref.mime_type,
        )
        return dict(result.metadata)

    def _enrich_parsed_metadata(
        self,
        file_ref: FileReference,
        parse_metadata: dict[str, Any],
    ) -> None:
        """Enrich FileReference with parsed metadata.

        This mirrors the logic in PipelineExecutor._enrich_parsed_metadata()
        for consistency.

        Args:
            file_ref: File reference to enrich
            parse_metadata: Metadata from parser
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

        if "parsed" not in file_ref.metadata:
            file_ref.metadata["parsed"] = {}

        for key, value in parse_metadata.items():
            if key in parsed_fields:
                file_ref.metadata["parsed"][key] = value


__all__ = [
    "PipelinePreviewService",
]
