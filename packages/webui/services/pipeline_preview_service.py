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
        path: list[str] = [SOURCE_NODE]

        # Evaluate entry routing (_source -> first node)
        entry_stage = self._evaluate_entry_routing(dag_obj, router, file_ref)
        routing_stages.append(entry_stage)

        if entry_stage.selected_node is None:
            # No route found
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            return RoutePreviewResponse(
                file_info=self._build_file_info(file_ref),
                sniff_result=sniff_dict if sniff_dict else None,
                routing_stages=routing_stages,
                path=path,
                parsed_metadata=None,
                total_duration_ms=total_duration_ms,
                warnings=warnings + ["No matching route found from source"],
            )

        # 5. Walk through the pipeline
        current_node = router.get_node(entry_stage.selected_node)
        path.append(entry_stage.selected_node)
        parsed_metadata: dict[str, Any] | None = None

        while current_node is not None:
            # Handle parser stage
            if current_node.type == NodeType.PARSER and include_parser_metadata:
                try:
                    parsed_metadata = self._run_parser(current_node, file_content, file_ref)
                    # Enrich file_ref with parsed metadata for mid-pipeline routing
                    self._enrich_parsed_metadata(file_ref, parsed_metadata)
                except Exception as e:
                    parser_id = current_node.plugin_id
                    filename = file_ref.filename or file_ref.uri
                    error_type = type(e).__name__

                    warnings.append(
                        f"Parser '{parser_id}' failed on '{filename}': {error_type}: {e}"
                    )
                    logger.warning(
                        "Parser %s failed during preview for %s: %s",
                        parser_id,
                        filename,
                        e,
                        exc_info=True,
                    )

            # Evaluate next routing stage
            next_stage = self._evaluate_next_routing(dag_obj, router, current_node, file_ref)
            if next_stage is not None:
                routing_stages.append(next_stage)

                if next_stage.selected_node is not None:
                    path.append(next_stage.selected_node)
                    current_node = router.get_node(next_stage.selected_node)
                else:
                    # No next node (terminal or no match)
                    break
            else:
                # No outgoing edges (terminal node like embedder)
                break

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        return RoutePreviewResponse(
            file_info=self._build_file_info(file_ref),
            sniff_result=sniff_dict if sniff_dict else None,
            routing_stages=routing_stages,
            path=path,
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
        # Extract extension
        path = Path(filename)
        extension = path.suffix.lower() if path.suffix else None

        # Guess MIME type
        mime_type, _ = mimetypes.guess_type(filename)

        # Compute content hash
        content_hash = hashlib.sha256(content).hexdigest()

        return FileReference(
            uri=f"preview://{filename}",
            source_type="preview",
            content_type="document",
            filename=filename,
            extension=extension,
            mime_type=mime_type,
            size_bytes=len(content),
            change_hint=content_hash,
            metadata={"source": {"filename": filename, "preview": True}},
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
        """Evaluate routing from _source to entry node.

        Args:
            dag: Pipeline DAG
            router: Pipeline router
            file_ref: File reference

        Returns:
            StageEvaluationResult with edge evaluations
        """
        # Get all edges from _source
        source_edges = [e for e in dag.edges if e.from_node == SOURCE_NODE]

        # Separate predicate edges from catch-all edges
        predicate_edges: list[PipelineEdge] = []
        catchall_edges: list[PipelineEdge] = []

        for edge in source_edges:
            if edge.when is None or edge.when == {}:
                catchall_edges.append(edge)
            else:
                predicate_edges.append(edge)

        evaluated_edges: list[EdgeEvaluationResult] = []
        selected_node: str | None = None
        found_match = False

        # Evaluate predicate edges first (in order)
        for edge in predicate_edges:
            if found_match:
                # Already found a match, mark as skipped
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=edge.when,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                    )
                )
            else:
                # Evaluate this edge
                field_evals = self._evaluate_predicate_fields(file_ref, edge.when)
                matched = all(fe.matched for fe in field_evals)

                if matched:
                    selected_node = edge.to_node
                    found_match = True
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
                    )
                )

        # Evaluate catch-all edges
        for edge in catchall_edges:
            if found_match:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                    )
                )
            else:
                # Catch-all always matches
                selected_node = edge.to_node
                found_match = True
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=True,
                        status="matched",
                        field_evaluations=None,
                    )
                )

        return StageEvaluationResult(
            stage="entry_routing",
            from_node=SOURCE_NODE,
            evaluated_edges=evaluated_edges,
            selected_node=selected_node,
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

        # Separate predicate edges from catch-all edges
        predicate_edges: list[PipelineEdge] = []
        catchall_edges: list[PipelineEdge] = []

        for edge in outgoing_edges:
            if edge.when is None or edge.when == {}:
                catchall_edges.append(edge)
            else:
                predicate_edges.append(edge)

        evaluated_edges: list[EdgeEvaluationResult] = []
        selected_node: str | None = None
        found_match = False

        # Evaluate predicate edges first
        for edge in predicate_edges:
            if found_match:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=edge.when,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                    )
                )
            else:
                field_evals = self._evaluate_predicate_fields(file_ref, edge.when)
                matched = all(fe.matched for fe in field_evals)

                if matched:
                    selected_node = edge.to_node
                    found_match = True
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
                    )
                )

        # Evaluate catch-all edges
        for edge in catchall_edges:
            if found_match:
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=False,
                        status="skipped",
                        field_evaluations=None,
                    )
                )
            else:
                selected_node = edge.to_node
                found_match = True
                evaluated_edges.append(
                    EdgeEvaluationResult(
                        from_node=edge.from_node,
                        to_node=edge.to_node,
                        predicate=None,
                        matched=True,
                        status="matched",
                        field_evaluations=None,
                    )
                )

        stage_name = f"{current_node.type.value}_to_next"
        return StageEvaluationResult(
            stage=stage_name,
            from_node=current_node.id,
            evaluated_edges=evaluated_edges,
            selected_node=selected_node,
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
