"""Source analysis tools for the SourceAnalyzer sub-agent.

These tools enable the SourceAnalyzer to investigate data sources:
- Enumerate files and understand source composition
- Sample files by various criteria
- Try different parsers on sample files
- Detect content language
- Preview file content
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

from webui.services.agent.tools.base import BaseTool

if TYPE_CHECKING:
    from shared.pipeline.types import FileReference

logger = logging.getLogger(__name__)

# Maximum files to enumerate before summarizing
MAX_ENUMERATE_FILES = 10000

# Maximum sample size
MAX_SAMPLE_SIZE = 100

# Maximum preview size in bytes/chars
MAX_PREVIEW_BYTES = 2048


class EnumerateFilesTool(BaseTool):
    """Enumerate files from a source and return composition statistics.

    Lists all files in a source and returns summary statistics including:
    - Total file count and size
    - Breakdown by file extension
    - Size distribution
    - Sample URIs for each type
    """

    NAME: ClassVar[str] = "enumerate_files"
    DESCRIPTION: ClassVar[str] = (
        "Enumerate all files from the configured source and return summary statistics. "
        "Returns counts and sizes by file extension, plus sample URIs for each type."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "include_samples": {
                "type": "boolean",
                "description": "Include sample URIs for each file type (default true)",
                "default": True,
            },
            "samples_per_type": {
                "type": "integer",
                "description": "Number of sample URIs to include per file type (default 3)",
                "default": 3,
            },
        },
        "required": [],
    }

    async def execute(
        self,
        include_samples: bool = True,
        samples_per_type: int = 3,
    ) -> dict[str, Any]:
        """Execute file enumeration and return statistics.

        Args:
            include_samples: Whether to include sample URIs
            samples_per_type: Number of samples per file type

        Returns:
            Dictionary with file statistics and optional samples
        """
        try:
            connector = self.context.get("connector")
            source_id = self.context.get("source_id")

            if not connector:
                return {
                    "success": False,
                    "error": "No connector available in context",
                }

            # Authenticate if needed
            if hasattr(connector, "authenticate"):
                auth_result = await connector.authenticate()
                if not auth_result:
                    return {
                        "success": False,
                        "error": "Failed to authenticate with source",
                    }

            # Collect statistics
            total_files = 0
            total_size = 0
            by_extension: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "total_size": 0, "samples": []})
            size_buckets = {
                "tiny": 0,  # < 1KB
                "small": 0,  # 1KB - 100KB
                "medium": 0,  # 100KB - 1MB
                "large": 0,  # 1MB - 10MB
                "huge": 0,  # > 10MB
            }

            # Store all file refs for potential sampling
            all_refs: list[FileReference] = []

            async for file_ref in connector.enumerate(source_id=source_id):
                total_files += 1
                total_size += file_ref.size_bytes

                ext = file_ref.extension.lower() if file_ref.extension else "(no ext)"
                by_extension[ext]["count"] += 1
                by_extension[ext]["total_size"] += file_ref.size_bytes

                # Collect samples
                if include_samples and len(by_extension[ext]["samples"]) < samples_per_type:
                    by_extension[ext]["samples"].append(file_ref.uri)

                # Size distribution
                size_bytes = file_ref.size_bytes
                if size_bytes < 1024:
                    size_buckets["tiny"] += 1
                elif size_bytes < 100 * 1024:
                    size_buckets["small"] += 1
                elif size_bytes < 1024 * 1024:
                    size_buckets["medium"] += 1
                elif size_bytes < 10 * 1024 * 1024:
                    size_buckets["large"] += 1
                else:
                    size_buckets["huge"] += 1

                # Keep reference for sampling
                if total_files <= MAX_ENUMERATE_FILES:
                    all_refs.append(file_ref)

                # Safety limit
                if total_files >= MAX_ENUMERATE_FILES:
                    logger.warning(
                        f"Enumeration limit reached ({MAX_ENUMERATE_FILES} files). Results may be incomplete."
                    )
                    break

            # Store enumerated refs in context for sampling tool
            self.context["_enumerated_files"] = all_refs

            # Format extension stats
            extension_stats = {}
            for ext, data in by_extension.items():
                extension_stats[ext] = {
                    "count": data["count"],
                    "total_size_bytes": data["total_size"],
                    "avg_size_bytes": data["total_size"] // data["count"] if data["count"] else 0,
                }
                if include_samples:
                    extension_stats[ext]["sample_uris"] = data["samples"]

            return {
                "success": True,
                "total_files": total_files,
                "total_size_bytes": total_size,
                "truncated": total_files >= MAX_ENUMERATE_FILES,
                "by_extension": extension_stats,
                "size_distribution": size_buckets,
                "extension_count": len(by_extension),
            }

        except Exception as e:
            logger.error(f"Failed to enumerate files: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


class SampleFilesTool(BaseTool):
    """Sample files from the source matching specific criteria.

    Allows intelligent sampling by:
    - File extension
    - Size range
    - Random selection
    - Content type
    """

    NAME: ClassVar[str] = "sample_files"
    DESCRIPTION: ClassVar[str] = (
        "Get a sample of files from the source matching specific criteria. "
        "Must run enumerate_files first. Supports filtering by extension, size, and random selection."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of files to sample (default 10, max 100)",
                "default": 10,
            },
            "extension": {
                "type": "string",
                "description": "Filter by file extension (e.g., '.pdf', '.md')",
            },
            "min_size_bytes": {
                "type": "integer",
                "description": "Minimum file size in bytes",
            },
            "max_size_bytes": {
                "type": "integer",
                "description": "Maximum file size in bytes",
            },
            "random": {
                "type": "boolean",
                "description": "Randomly shuffle results (default true)",
                "default": True,
            },
            "content_type": {
                "type": "string",
                "description": "Filter by content_type field (e.g., 'file', 'message')",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        count: int = 10,
        extension: str | None = None,
        min_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
        random_shuffle: bool = True,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Execute file sampling.

        Args:
            count: Number of files to return
            extension: Filter by extension
            min_size_bytes: Minimum size filter
            max_size_bytes: Maximum size filter
            random_shuffle: Whether to randomize results
            content_type: Filter by content type

        Returns:
            Dictionary with sampled file references
        """
        # Handle the 'random' parameter name collision with Python's random module
        if "random" in self.context.get("_tool_kwargs", {}):
            random_shuffle = self.context["_tool_kwargs"]["random"]

        try:
            # Check for enumerated files
            enumerated = self.context.get("_enumerated_files", [])
            if not enumerated:
                return {
                    "success": False,
                    "error": "No files enumerated. Run enumerate_files first.",
                }

            # Apply filters
            filtered = enumerated

            if extension:
                ext = extension.lower() if not extension.startswith(".") else extension.lower()
                if not ext.startswith("."):
                    ext = f".{ext}"
                filtered = [f for f in filtered if f.extension.lower() == ext]

            if min_size_bytes is not None:
                filtered = [f for f in filtered if f.size_bytes >= min_size_bytes]

            if max_size_bytes is not None:
                filtered = [f for f in filtered if f.size_bytes <= max_size_bytes]

            if content_type:
                filtered = [f for f in filtered if f.content_type == content_type]

            # Randomize if requested
            if random_shuffle:
                filtered = list(filtered)
                random.shuffle(filtered)

            # Limit count
            count = min(count, MAX_SAMPLE_SIZE)
            sampled = filtered[:count]

            # Format output
            files = []
            for ref in sampled:
                files.append(
                    {
                        "uri": ref.uri,
                        "filename": ref.filename,
                        "extension": ref.extension,
                        "mime_type": ref.mime_type,
                        "size_bytes": ref.size_bytes,
                        "content_type": ref.content_type,
                        "change_hint": ref.change_hint,
                    }
                )

            return {
                "success": True,
                "count": len(files),
                "total_matching": len(filtered),
                "files": files,
                "filters_applied": {
                    "extension": extension,
                    "min_size_bytes": min_size_bytes,
                    "max_size_bytes": max_size_bytes,
                    "content_type": content_type,
                },
            }

        except Exception as e:
            logger.error(f"Failed to sample files: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


class TryParserTool(BaseTool):
    """Try parsing a file with a specific parser plugin.

    Attempts to parse a file and returns success/failure along with
    basic statistics about the parsed content.
    """

    NAME: ClassVar[str] = "try_parser"
    DESCRIPTION: ClassVar[str] = (
        "Attempt to parse a file with a specific parser plugin. "
        "Returns success/failure and statistics about the parsed content "
        "(text length, structure info, etc.)."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of the file to parse (from sample_files)",
            },
            "parser_id": {
                "type": "string",
                "description": "ID of the parser plugin to use (e.g., 'text', 'unstructured')",
            },
            "config": {
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
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute file parsing.

        Args:
            file_uri: URI of the file to parse
            parser_id: Parser plugin ID
            config: Optional parser config overrides

        Returns:
            Dictionary with parse results and statistics
        """
        try:
            from shared.plugins.loader import load_plugins
            from shared.plugins.registry import plugin_registry

            # Find the file reference
            enumerated = self.context.get("_enumerated_files", [])
            file_ref = next((f for f in enumerated if f.uri == file_uri), None)

            if not file_ref:
                return {
                    "success": False,
                    "error": f"File not found in enumerated files: {file_uri}",
                }

            # Load content from connector
            connector = self.context.get("connector")
            if not connector:
                return {
                    "success": False,
                    "error": "No connector available to load content",
                }

            content = await connector.load_content(file_ref)

            # Get parser plugin
            load_plugins(plugin_types={"parser"})
            parser_record = plugin_registry.get("parser", parser_id)

            if not parser_record:
                available = list(plugin_registry.get_by_type("parser").keys())
                return {
                    "success": False,
                    "error": f"Parser '{parser_id}' not found",
                    "available_parsers": available,
                }

            # Create parser instance
            parser_config = config or {}
            parser = parser_record.plugin_class(parser_config)

            # Parse the content
            parse_result = await parser.parse(content, file_ref)

            # Collect statistics
            text = parse_result.text if hasattr(parse_result, "text") else str(parse_result)
            text_length = len(text)
            line_count = text.count("\n") + 1
            word_count = len(text.split())

            # Check for common content patterns
            has_headers = any(text.startswith(h) for h in ["#", "##", "###"]) or "\n#" in text
            has_code_blocks = "```" in text or "    " in text[:1000]
            has_tables = "|" in text and "-|-" in text

            return {
                "success": True,
                "file_uri": file_uri,
                "parser_id": parser_id,
                "stats": {
                    "text_length": text_length,
                    "line_count": line_count,
                    "word_count": word_count,
                },
                "content_hints": {
                    "has_headers": has_headers,
                    "has_code_blocks": has_code_blocks,
                    "has_tables": has_tables,
                },
                "parse_metadata": (parse_result.parse_metadata if hasattr(parse_result, "parse_metadata") else {}),
                "preview": text[:500] if text else "",
            }

        except (NotImplementedError, AttributeError):
            # Parser plugin interface not fully implemented yet
            # Fall back to basic text extraction
            return await self._fallback_parse(file_uri, parser_id)

        except Exception as e:
            logger.error(f"Failed to parse file {file_uri} with {parser_id}: {e}", exc_info=True)
            return {
                "success": False,
                "file_uri": file_uri,
                "parser_id": parser_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _fallback_parse(
        self,
        file_uri: str,
        parser_id: str,
    ) -> dict[str, Any]:
        """Fallback parsing for when parser plugins aren't fully implemented."""
        try:
            connector = self.context.get("connector")
            enumerated = self.context.get("_enumerated_files", [])
            file_ref = next((f for f in enumerated if f.uri == file_uri), None)

            if not file_ref or not connector:
                return {
                    "success": False,
                    "error": "Cannot perform fallback parse",
                }

            content = await connector.load_content(file_ref)

            # Try to decode as text
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1")
                except UnicodeDecodeError:
                    return {
                        "success": False,
                        "file_uri": file_uri,
                        "parser_id": parser_id,
                        "error": "Binary content - cannot parse as text",
                    }

            return {
                "success": True,
                "file_uri": file_uri,
                "parser_id": parser_id,
                "stats": {
                    "text_length": len(text),
                    "line_count": text.count("\n") + 1,
                    "word_count": len(text.split()),
                },
                "content_hints": {
                    "has_headers": "#" in text[:1000],
                    "has_code_blocks": "```" in text or "    " in text[:1000],
                    "has_tables": "|" in text and "-" in text,
                },
                "parse_metadata": {"fallback": True},
                "preview": text[:500],
            }

        except Exception as e:
            return {
                "success": False,
                "file_uri": file_uri,
                "parser_id": parser_id,
                "error": f"Fallback parse failed: {e}",
            }


class DetectLanguageTool(BaseTool):
    """Detect the language of text content.

    Uses language detection to identify the primary language(s)
    present in text content.
    """

    NAME: ClassVar[str] = "detect_language"
    DESCRIPTION: ClassVar[str] = (
        "Detect the language of text content. Can analyze a file's content "
        "or provided text directly. Returns detected language(s) with confidence."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of file to analyze (from sample_files)",
            },
            "text": {
                "type": "string",
                "description": "Text to analyze directly (alternative to file_uri)",
            },
            "sample_size": {
                "type": "integer",
                "description": "Characters to sample for detection (default 5000)",
                "default": 5000,
            },
        },
        "required": [],
    }

    async def execute(
        self,
        file_uri: str | None = None,
        text: str | None = None,
        sample_size: int = 5000,
    ) -> dict[str, Any]:
        """Execute language detection.

        Args:
            file_uri: File to analyze
            text: Text to analyze directly
            sample_size: Characters to sample

        Returns:
            Dictionary with detected language(s)
        """
        try:
            # Get text to analyze
            if text:
                content_text = text
            elif file_uri:
                connector = self.context.get("connector")
                enumerated = self.context.get("_enumerated_files", [])
                file_ref = next((f for f in enumerated if f.uri == file_uri), None)

                if not file_ref or not connector:
                    return {
                        "success": False,
                        "error": f"File not found: {file_uri}",
                    }

                content = await connector.load_content(file_ref)
                try:
                    content_text = content.decode("utf-8")
                except UnicodeDecodeError:
                    content_text = content.decode("latin-1", errors="replace")
            else:
                return {
                    "success": False,
                    "error": "Either file_uri or text must be provided",
                }

            # Sample the text
            sampled = content_text[:sample_size]

            # Try to use langdetect if available
            try:
                from langdetect import detect_langs

                detected = detect_langs(sampled)
                languages = [{"code": lang.lang, "confidence": round(lang.prob, 3)} for lang in detected[:3]]  # Top 3
                primary = detected[0].lang if detected else "unknown"

            except ImportError:
                # Fallback: simple heuristic-based detection
                languages, primary = self._heuristic_detect(sampled)

            return {
                "success": True,
                "primary_language": primary,
                "detected_languages": languages,
                "sample_length": len(sampled),
                "source": file_uri if file_uri else "provided_text",
            }

        except Exception as e:
            logger.error(f"Failed to detect language: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _heuristic_detect(self, text: str) -> tuple[list[dict[str, Any]], str]:
        """Simple heuristic language detection fallback."""
        # Common language indicators
        indicators = {
            "en": ["the ", " and ", " of ", " to ", " is ", " in "],
            "es": [" de ", " la ", " el ", " en ", " que ", " es "],
            "fr": [" de ", " le ", " la ", " et ", " en ", " est "],
            "de": [" der ", " die ", " und ", " in ", " ist ", " den "],
            "zh": ["的", "是", "在", "了", "我", "有"],
            "ja": ["の", "は", "が", "を", "に", "た"],
        }

        text_lower = text.lower()
        scores: dict[str, int] = {}

        for lang, patterns in indicators.items():
            score = sum(1 for p in patterns if p in text_lower)
            if score > 0:
                scores[lang] = score

        if not scores:
            return [{"code": "unknown", "confidence": 0.0}], "unknown"

        # Sort by score
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in sorted_langs)

        languages = [{"code": lang, "confidence": round(score / total, 3)} for lang, score in sorted_langs[:3]]

        return languages, sorted_langs[0][0]


class GetFileContentPreviewTool(BaseTool):
    """Get a preview of file content for inspection.

    Returns the first N bytes or characters of a file for
    human or agent inspection of content structure.
    """

    NAME: ClassVar[str] = "get_file_content_preview"
    DESCRIPTION: ClassVar[str] = (
        "Get a preview of a file's content. Returns the first N bytes/chars "
        "for inspection. Useful for understanding content structure."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_uri": {
                "type": "string",
                "description": "URI of file to preview (from sample_files)",
            },
            "max_bytes": {
                "type": "integer",
                "description": f"Maximum bytes to return (default {MAX_PREVIEW_BYTES})",
                "default": MAX_PREVIEW_BYTES,
            },
            "as_text": {
                "type": "boolean",
                "description": "Try to decode as text (default true)",
                "default": True,
            },
        },
        "required": ["file_uri"],
    }

    async def execute(
        self,
        file_uri: str,
        max_bytes: int = MAX_PREVIEW_BYTES,
        as_text: bool = True,
    ) -> dict[str, Any]:
        """Execute content preview.

        Args:
            file_uri: URI of file to preview
            max_bytes: Maximum bytes to return
            as_text: Whether to decode as text

        Returns:
            Dictionary with content preview
        """
        try:
            connector = self.context.get("connector")
            enumerated = self.context.get("_enumerated_files", [])
            file_ref = next((f for f in enumerated if f.uri == file_uri), None)

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

            # Limit size
            max_bytes = min(max_bytes, MAX_PREVIEW_BYTES)
            preview_bytes = content[:max_bytes]

            result: dict[str, Any] = {
                "success": True,
                "file_uri": file_uri,
                "filename": file_ref.filename,
                "mime_type": file_ref.mime_type,
                "total_size_bytes": file_ref.size_bytes,
                "preview_size_bytes": len(preview_bytes),
            }

            if as_text:
                # Try to decode as text
                try:
                    text = preview_bytes.decode("utf-8")
                    result["content_type"] = "text"
                    result["preview"] = text
                    result["encoding"] = "utf-8"
                except UnicodeDecodeError:
                    try:
                        text = preview_bytes.decode("latin-1")
                        result["content_type"] = "text"
                        result["preview"] = text
                        result["encoding"] = "latin-1"
                    except UnicodeDecodeError:
                        result["content_type"] = "binary"
                        result["preview"] = None
                        result["note"] = "Binary content - cannot display as text"
            else:
                result["content_type"] = "binary"
                result["preview"] = None
                result["preview_hex"] = preview_bytes[:256].hex()

            return result

        except Exception as e:
            logger.error(f"Failed to get content preview: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
