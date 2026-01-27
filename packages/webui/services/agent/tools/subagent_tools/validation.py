"""Pipeline validation tools for the PipelineValidator sub-agent.

These tools enable the PipelineValidator to test and validate pipelines:
- Run dry-run validation on sample files
- Inspect failure details
- Try alternative parser configurations
- Compare parser outputs
- Inspect chunk output
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from webui.services.agent.tools.base import BaseTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shared.pipeline.executor_types import ExecutionResult, SampleOutput, StageFailure
    from shared.pipeline.types import FileReference

logger = logging.getLogger(__name__)

# Maximum files to test in a single dry-run
MAX_DRY_RUN_FILES = 100

# Maximum chunk previews to return
MAX_CHUNK_PREVIEWS = 5

# Preview length for chunk content
CHUNK_PREVIEW_LENGTH = 200


class RunDryRunTool(BaseTool):
    """Run pipeline in dry-run mode on sample files.

    Executes the pipeline without writing to storage, validating that
    files can be successfully processed through all stages.
    """

    NAME: ClassVar[str] = "run_dry_run"
    DESCRIPTION: ClassVar[str] = (
        "Run the pipeline in dry-run mode on sample files. Returns success/failure "
        "counts, success rate, stage timings, and chunk statistics. Use this to "
        "validate a pipeline configuration before applying it."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pipeline": {
                "type": "object",
                "description": "Pipeline DAG configuration to validate",
            },
            "sample_count": {
                "type": "integer",
                "description": "Number of files to test (default 50, max 100)",
                "default": 50,
            },
        },
        "required": ["pipeline"],
    }

    async def execute(
        self,
        pipeline: dict[str, Any],
        sample_count: int = 50,
    ) -> dict[str, Any]:
        """Execute dry-run validation.

        Args:
            pipeline: Pipeline DAG configuration
            sample_count: Number of files to test

        Returns:
            Dictionary with validation results
        """
        try:
            from shared.pipeline.executor import PipelineExecutor
            from shared.pipeline.executor_types import ExecutionMode
            from shared.pipeline.types import PipelineDAG

            # Get context
            session = self.context.get("session")
            connector = self.context.get("connector")
            sample_files: list[FileReference] = self.context.get("sample_files", [])

            if not session:
                return {
                    "success": False,
                    "error": "No database session available",
                }

            if not connector:
                return {
                    "success": False,
                    "error": "No connector available",
                }

            if not sample_files:
                # Try to get from enumerated files
                sample_files = self.context.get("_enumerated_files", [])

            if not sample_files:
                return {
                    "success": False,
                    "error": "No sample files available. Run enumerate_files or provide sample_files.",
                }

            # Limit sample count
            sample_count = min(sample_count, MAX_DRY_RUN_FILES, len(sample_files))
            files_to_test = sample_files[:sample_count]

            # Parse pipeline DAG
            try:
                dag = PipelineDAG.from_dict(pipeline)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid pipeline configuration: {e}",
                }

            # Validate DAG with known plugins for complete validation
            # Note: plugin_registry may be empty if plugins weren't loaded at startup.
            # Load the pipeline-critical plugin types on-demand to avoid false failures.
            from shared.plugins.loader import load_plugins
            from shared.plugins.registry import plugin_registry

            if not plugin_registry.list_ids():
                load_plugins(plugin_types={"parser", "chunking", "extractor", "embedding"})

            known_plugins = set(plugin_registry.list_ids())
            errors = dag.validate(known_plugins)
            if errors:
                return {
                    "success": False,
                    "error": "Pipeline validation failed",
                    "validation_errors": [{"rule": e.rule, "message": e.message, "node_id": e.node_id} for e in errors],
                }

            # Create a mock collection ID for dry-run
            collection_id = "dry-run-validation"

            # Create executor
            executor = PipelineExecutor(
                dag=dag,
                collection_id=collection_id,
                session=session,
                connector=connector,
                mode=ExecutionMode.DRY_RUN,
            )

            # Create async iterator from files
            async def file_iterator() -> AsyncIterator[FileReference]:
                for f in files_to_test:
                    yield f

            # Run dry-run
            result: ExecutionResult = await executor.execute(
                file_refs=file_iterator(),
                limit=sample_count,
            )

            # Calculate success rate
            total_tested = result.files_succeeded + result.files_failed
            success_rate = result.files_succeeded / total_tested if total_tested > 0 else 0.0

            # Store result in context for other tools
            self.context["_dry_run_result"] = result
            self.context["_dry_run_failures"] = {f.file_uri: f for f in result.failures}

            # Build response
            response: dict[str, Any] = {
                "success": True,
                "files_tested": total_tested,
                "files_succeeded": result.files_succeeded,
                "files_failed": result.files_failed,
                "files_skipped": result.files_skipped,
                "success_rate": round(success_rate, 4),
                "stage_timings": result.stage_timings,
                "total_duration_ms": round(result.total_duration_ms, 2),
            }

            if result.chunk_stats:
                response["chunk_stats"] = result.chunk_stats.to_dict()

            # Categorize failures
            if result.failures:
                failure_categories: dict[str, list[str]] = defaultdict(list)
                for failure in result.failures:
                    category = self._categorize_failure(failure)
                    failure_categories[category].append(failure.file_uri)

                response["failure_categories"] = {
                    cat: {"count": len(uris), "example_files": uris[:3]} for cat, uris in failure_categories.items()
                }

            # Include sample output URIs
            if result.sample_outputs:
                response["sample_output_uris"] = [s.file_ref.uri for s in result.sample_outputs[:10]]
                # Store for inspection
                self.context["_sample_outputs"] = {s.file_ref.uri: s for s in result.sample_outputs}

            # Assessment
            if success_rate >= 0.95:
                response["assessment"] = "ready"
                response["assessment_message"] = "Pipeline is ready for production use."
            elif success_rate >= 0.90:
                response["assessment"] = "needs_review"
                response["assessment_message"] = "Some files failed. Review failures to decide if acceptable."
            else:
                response["assessment"] = "blocking_issues"
                response["assessment_message"] = "Too many failures. Investigate and adjust pipeline configuration."

            return response

        except Exception as e:
            logger.error(f"Dry-run failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _categorize_failure(self, failure: StageFailure) -> str:
        """Categorize a failure based on error type and message."""
        error_type = failure.error_type.lower()
        error_msg = failure.error_message.lower()

        if "parse" in error_type or "parse" in error_msg:
            return "parser_error"
        if "unicode" in error_type or "decode" in error_msg:
            return "encoding_error"
        if "timeout" in error_type or "timeout" in error_msg:
            return "timeout"
        if "memory" in error_type or "oom" in error_msg:
            return "memory_error"
        if "not found" in error_msg or "missing" in error_msg:
            return "file_not_found"
        if "permission" in error_msg or "access" in error_msg:
            return "permission_error"
        if "corrupt" in error_msg or "invalid" in error_msg:
            return "corrupted_file"
        if failure.stage_type == "loader":
            return "load_error"
        if failure.stage_type == "parser":
            return "parser_error"
        if failure.stage_type == "chunker":
            return "chunker_error"

        return "unknown_error"


class GetFailureDetailsTool(BaseTool):
    """Get detailed error information for a specific failure.

    Retrieves the full error details including traceback for a file
    that failed during dry-run validation.
    """

    NAME: ClassVar[str] = "get_failure_details"
    DESCRIPTION: ClassVar[str] = (
        "Get detailed error information for a specific file that failed during "
        "dry-run validation. Returns stage, error type, message, and traceback."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of the file to get failure details for",
            },
        },
        "required": ["file_uri"],
    }

    async def execute(self, file_uri: str) -> dict[str, Any]:
        """Get failure details for a file.

        Args:
            file_uri: URI of the failed file

        Returns:
            Dictionary with failure details
        """
        try:
            failures = self.context.get("_dry_run_failures", {})

            if not failures:
                return {
                    "success": False,
                    "error": "No dry-run results available. Run run_dry_run first.",
                }

            failure: StageFailure | None = failures.get(file_uri)

            if not failure:
                # Check if file was successful instead
                result = self.context.get("_dry_run_result")
                if result and result.sample_outputs:
                    for sample in result.sample_outputs:
                        if sample.file_ref.uri == file_uri:
                            return {
                                "success": True,
                                "file_uri": file_uri,
                                "status": "succeeded",
                                "message": "This file succeeded during dry-run",
                            }

                return {
                    "success": False,
                    "error": f"No failure found for: {file_uri}",
                    "available_failures": list(failures.keys())[:10],
                }

            return {
                "success": True,
                "file_uri": failure.file_uri,
                "stage_id": failure.stage_id,
                "stage_type": failure.stage_type,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
                "error_traceback": failure.error_traceback,
                "category": self._categorize_failure(failure),
            }

        except Exception as e:
            logger.error(f"Failed to get failure details: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _categorize_failure(self, failure: StageFailure) -> str:
        """Categorize the failure for easier understanding."""
        error_type = failure.error_type.lower()
        error_msg = failure.error_message.lower()

        if "parse" in error_type or "parse" in error_msg:
            return "parser_error"
        if "unicode" in error_type or "decode" in error_msg:
            return "encoding_error"
        if failure.stage_type == "loader":
            return "load_error"

        return "processing_error"


class TryAlternativeConfigTool(BaseTool):
    """Try processing a file with alternative parser configuration.

    Re-runs a failed file with different parser or config settings
    to see if an alternative approach works.
    """

    NAME: ClassVar[str] = "try_alternative_config"
    DESCRIPTION: ClassVar[str] = (
        "Try processing a failed file with an alternative parser or configuration. "
        "Useful for finding workarounds for specific file types or formats."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of the file to retry",
            },
            "parser_id": {
                "type": "string",
                "description": "Alternative parser plugin ID to try",
            },
            "parser_config": {
                "type": "object",
                "description": "Optional parser configuration overrides",
            },
        },
        "required": ["file_uri", "parser_id"],
    }

    async def execute(
        self,
        file_uri: str,
        parser_id: str,
        parser_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Try alternative parser configuration.

        Args:
            file_uri: URI of the file to retry
            parser_id: Alternative parser ID
            parser_config: Optional config overrides

        Returns:
            Dictionary with retry results
        """
        try:
            from shared.pipeline.executor import PipelineExecutor
            from shared.pipeline.executor_types import ExecutionMode
            from shared.pipeline.types import (
                NodeType,
                PipelineDAG,
                PipelineEdge,
                PipelineNode,
            )

            # Get context
            session = self.context.get("session")
            connector = self.context.get("connector")
            enumerated = self.context.get("_enumerated_files", [])
            sample_files = self.context.get("sample_files", [])

            # Find the file reference
            all_files = enumerated + sample_files
            file_ref = next((f for f in all_files if f.uri == file_uri), None)

            if not file_ref:
                return {
                    "success": False,
                    "error": f"File not found: {file_uri}",
                }

            if not session:
                return {
                    "success": False,
                    "error": "No database session available",
                }

            # Check if parser exists
            from shared.plugins import plugin_registry
            from shared.plugins.loader import load_plugins

            load_plugins(plugin_types={"parser"})
            parser_record = plugin_registry.get("parser", parser_id)

            if not parser_record:
                available = list(plugin_registry.get_by_type("parser").keys())
                return {
                    "success": False,
                    "error": f"Parser '{parser_id}' not found",
                    "available_parsers": available,
                }

            # Build a simple single-parser pipeline DAG for testing
            config = parser_config or {}
            dag = PipelineDAG(
                id="test-alternative",
                version="1",
                nodes=[
                    PipelineNode(
                        id="alt_parser",
                        type=NodeType.PARSER,
                        plugin_id=parser_id,
                        config=config,
                    ),
                    PipelineNode(
                        id="chunker",
                        type=NodeType.CHUNKER,
                        plugin_id="character",
                        config={"max_tokens": 1000},
                    ),
                    PipelineNode(
                        id="embedder",
                        type=NodeType.EMBEDDER,
                        plugin_id="mock",
                        config={},
                    ),
                ],
                edges=[
                    PipelineEdge(from_node="_source", to_node="alt_parser", when=None),
                    PipelineEdge(from_node="alt_parser", to_node="chunker"),
                    PipelineEdge(from_node="chunker", to_node="embedder"),
                ],
            )

            # Create executor for single file
            executor = PipelineExecutor(
                dag=dag,
                collection_id="test-alternative",
                session=session,
                connector=connector,
                mode=ExecutionMode.DRY_RUN,
            )

            # Run on single file - use async generator with proper typing
            async def single_file_iterator() -> AsyncIterator[FileReference]:
                yield file_ref

            start_time = time.perf_counter()
            result = await executor.execute(
                file_refs=single_file_iterator(),
                limit=1,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Check original failure
            original_failure = self.context.get("_dry_run_failures", {}).get(file_uri)

            # Build response
            response: dict[str, Any] = {
                "success": True,
                "file_uri": file_uri,
                "parser_id": parser_id,
                "parser_config": config,
                "duration_ms": round(duration_ms, 2),
            }

            if result.files_succeeded > 0:
                response["parse_success"] = True
                response["message"] = f"Alternative parser '{parser_id}' succeeded!"

                if result.sample_outputs:
                    sample = result.sample_outputs[0]
                    response["chunks_created"] = len(sample.chunks)
                    response["parse_metadata"] = sample.parse_metadata

                if original_failure:
                    response["comparison"] = {
                        "original_failed": True,
                        "original_parser": original_failure.stage_id,
                        "original_error": original_failure.error_type,
                    }
            else:
                response["parse_success"] = False

                if result.failures:
                    failure = result.failures[0]
                    response["error_type"] = failure.error_type
                    response["error_message"] = failure.error_message
                    response["message"] = f"Alternative parser '{parser_id}' also failed"

            return response

        except Exception as e:
            logger.error(f"Failed to try alternative config: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


class CompareParserOutputTool(BaseTool):
    """Compare output of two parsers on the same file.

    Parses a file with two different parsers and compares the results,
    helping to choose the best parser for specific file types.
    """

    NAME: ClassVar[str] = "compare_parser_output"
    DESCRIPTION: ClassVar[str] = (
        "Compare the output of two parsers on the same file. Returns comparison "
        "of text length, word count, structure detection, and parsing time."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of the file to compare",
            },
            "parser_a": {
                "type": "string",
                "description": "First parser plugin ID",
            },
            "parser_b": {
                "type": "string",
                "description": "Second parser plugin ID",
            },
            "config_a": {
                "type": "object",
                "description": "Optional config for parser A",
            },
            "config_b": {
                "type": "object",
                "description": "Optional config for parser B",
            },
        },
        "required": ["file_uri", "parser_a", "parser_b"],
    }

    async def execute(
        self,
        file_uri: str,
        parser_a: str,
        parser_b: str,
        config_a: dict[str, Any] | None = None,
        config_b: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare two parsers on the same file.

        Args:
            file_uri: URI of file to parse
            parser_a: First parser ID
            parser_b: Second parser ID
            config_a: Config for parser A
            config_b: Config for parser B

        Returns:
            Dictionary with comparison results
        """
        try:
            from shared.plugins import plugin_registry
            from shared.plugins.loader import load_plugins

            # Get file content
            connector = self.context.get("connector")
            enumerated = self.context.get("_enumerated_files", [])
            sample_files = self.context.get("sample_files", [])

            all_files = enumerated + sample_files
            file_ref = next((f for f in all_files if f.uri == file_uri), None)

            if not file_ref:
                return {
                    "success": False,
                    "error": f"File not found: {file_uri}",
                }

            if not connector:
                return {
                    "success": False,
                    "error": "No connector available",
                }

            # Load content
            content = await connector.load_content(file_ref)

            # Load parsers
            load_plugins(plugin_types={"parser"})

            results: dict[str, dict[str, Any]] = {}

            for parser_id, config in [(parser_a, config_a), (parser_b, config_b)]:
                start_time = time.perf_counter()

                try:
                    record = plugin_registry.get("parser", parser_id)
                    if not record:
                        results[parser_id] = {
                            "success": False,
                            "error": f"Parser not found: {parser_id}",
                        }
                        continue

                    parser = record.plugin_class(config or {})
                    parse_result = parser.parse_bytes(
                        content,
                        filename=file_ref.filename,
                        file_extension=file_ref.extension,
                        mime_type=file_ref.mime_type,
                    )

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    text = parse_result.text

                    results[parser_id] = {
                        "success": True,
                        "text_length": len(text),
                        "line_count": text.count("\n") + 1,
                        "word_count": len(text.split()),
                        "has_headers": "#" in text[:1000] or "\n#" in text,
                        "has_code_blocks": "```" in text,
                        "has_tables": "|" in text and "-|-" in text,
                        "duration_ms": round(duration_ms, 2),
                        "metadata": dict(parse_result.metadata) if hasattr(parse_result, "metadata") else {},
                    }

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    results[parser_id] = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": round(duration_ms, 2),
                    }

            # Build comparison
            comparison: dict[str, Any] = {
                "success": True,
                "file_uri": file_uri,
                "parser_a": parser_a,
                "parser_b": parser_b,
                "results": results,
            }

            # Add comparison summary if both succeeded
            if results.get(parser_a, {}).get("success") and results.get(parser_b, {}).get("success"):
                a = results[parser_a]
                b = results[parser_b]

                comparison["comparison"] = {
                    "text_length_diff": a["text_length"] - b["text_length"],
                    "word_count_diff": a["word_count"] - b["word_count"],
                    "duration_diff_ms": a["duration_ms"] - b["duration_ms"],
                    "both_have_headers": a["has_headers"] and b["has_headers"],
                    "both_have_tables": a["has_tables"] and b["has_tables"],
                }

                # Recommendation
                if a["text_length"] > b["text_length"] * 1.1:
                    comparison["recommendation"] = parser_a
                    comparison["recommendation_reason"] = "Extracts more content"
                elif b["text_length"] > a["text_length"] * 1.1:
                    comparison["recommendation"] = parser_b
                    comparison["recommendation_reason"] = "Extracts more content"
                elif a["duration_ms"] < b["duration_ms"] * 0.5:
                    comparison["recommendation"] = parser_a
                    comparison["recommendation_reason"] = "Significantly faster"
                elif b["duration_ms"] < a["duration_ms"] * 0.5:
                    comparison["recommendation"] = parser_b
                    comparison["recommendation_reason"] = "Significantly faster"
                else:
                    comparison["recommendation"] = None
                    comparison["recommendation_reason"] = "Results are similar"

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare parsers: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


class InspectChunksTool(BaseTool):
    """Inspect chunk output for a specific file.

    Examines the chunks created during dry-run to verify chunking
    behavior and quality.
    """

    NAME: ClassVar[str] = "inspect_chunks"
    DESCRIPTION: ClassVar[str] = (
        "Examine chunks created during dry-run for a specific file. Returns chunk "
        "count, size statistics, and content previews to verify chunking quality."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of file to inspect chunks for",
            },
            "preview_count": {
                "type": "integer",
                "description": "Number of chunk previews to return (default 5)",
                "default": 5,
            },
        },
        "required": ["file_uri"],
    }

    async def execute(
        self,
        file_uri: str,
        preview_count: int = 5,
    ) -> dict[str, Any]:
        """Inspect chunks for a file.

        Args:
            file_uri: URI of file to inspect
            preview_count: Number of previews to return

        Returns:
            Dictionary with chunk inspection results
        """
        try:
            sample_outputs: dict[str, SampleOutput] = self.context.get("_sample_outputs", {})

            if not sample_outputs:
                return {
                    "success": False,
                    "error": "No sample outputs available. Run run_dry_run first.",
                }

            sample: SampleOutput | None = sample_outputs.get(file_uri)

            if not sample:
                return {
                    "success": False,
                    "error": f"No chunks found for: {file_uri}",
                    "available_files": list(sample_outputs.keys())[:10],
                }

            chunks = sample.chunks
            chunk_count = len(chunks)

            if chunk_count == 0:
                return {
                    "success": True,
                    "file_uri": file_uri,
                    "chunk_count": 0,
                    "warning": "File produced no chunks",
                }

            # Calculate statistics
            sizes = []
            token_counts = []

            for chunk in chunks:
                content = chunk.get("content") or chunk.get("text") or ""
                sizes.append(len(content))

                metadata = chunk.get("metadata", {})
                if "token_count" in metadata:
                    token_counts.append(metadata["token_count"])

            avg_size = sum(sizes) / len(sizes) if sizes else 0
            min_size = min(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0

            response: dict[str, Any] = {
                "success": True,
                "file_uri": file_uri,
                "chunk_count": chunk_count,
                "size_stats": {
                    "avg_chars": round(avg_size, 1),
                    "min_chars": min_size,
                    "max_chars": max_size,
                },
                "parse_metadata": sample.parse_metadata,
            }

            if token_counts:
                response["token_stats"] = {
                    "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
                    "min_tokens": min(token_counts),
                    "max_tokens": max(token_counts),
                }

            # Add previews
            preview_count = min(preview_count, MAX_CHUNK_PREVIEWS, chunk_count)
            previews = []

            for i, chunk in enumerate(chunks[:preview_count]):
                content = chunk.get("content") or chunk.get("text") or ""
                preview = content[:CHUNK_PREVIEW_LENGTH]
                if len(content) > CHUNK_PREVIEW_LENGTH:
                    preview += "..."

                previews.append(
                    {
                        "index": i,
                        "size_chars": len(content),
                        "preview": preview,
                        "metadata": chunk.get("metadata", {}),
                    }
                )

            response["previews"] = previews

            # Quality assessment
            if avg_size < 100:
                response["quality_warning"] = "Chunks are very small - may be over-chunked"
            elif avg_size > 5000:
                response["quality_warning"] = "Chunks are very large - may be under-chunked"

            if max_size > min_size * 10:
                response["quality_warning"] = "High variance in chunk sizes"

            return response

        except Exception as e:
            logger.error(f"Failed to inspect chunks: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


__all__ = [
    "RunDryRunTool",
    "GetFailureDetailsTool",
    "TryAlternativeConfigTool",
    "CompareParserOutputTool",
    "InspectChunksTool",
]
