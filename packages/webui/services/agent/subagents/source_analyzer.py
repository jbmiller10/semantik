"""SourceAnalyzer sub-agent for investigating data sources.

The SourceAnalyzer is a specialized sub-agent that investigates a data source
to understand its composition and recommend appropriate parsing strategies.

It systematically:
1. Enumerates files to understand source composition
2. Samples representative files from each major type
3. Tries parsing samples to verify they work
4. Detects patterns (scanned PDFs, mixed languages, code vs prose)
5. Notes uncertainties for the user

Returns a structured SourceAnalysis with recommendations for the orchestrator.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from webui.services.agent.subagents.base import (
    Message,
    SubAgent,
    SubAgentResult,
    Uncertainty,
)
from webui.services.agent.tools.subagent_tools.source import (
    DetectLanguageTool,
    EnumerateFilesTool,
    GetFileContentPreviewTool,
    SampleFilesTool,
    TryParserTool,
)

if TYPE_CHECKING:
    from webui.services.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class FileTypeStats:
    """Statistics for a file type."""

    count: int
    total_size_bytes: int
    sample_uris: list[str] = field(default_factory=list)
    parser_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ContentCharacteristics:
    """Detected content characteristics."""

    languages: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)  # academic, code, docs, etc.
    quality_issues: list[str] = field(default_factory=list)  # scanned PDFs, corrupted, etc.


@dataclass
class ParserRecommendation:
    """Recommended parser for a file type."""

    extension: str
    parser_id: str
    confidence: float  # 0.0 - 1.0
    notes: str | None = None


@dataclass
class SourceAnalysis:
    """Complete analysis of a data source."""

    total_files: int
    total_size_bytes: int
    by_extension: dict[str, FileTypeStats]
    content_characteristics: ContentCharacteristics
    parser_recommendations: list[ParserRecommendation]
    uncertainties: list[Uncertainty]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "by_extension": {
                ext: {
                    "count": stats.count,
                    "total_size_bytes": stats.total_size_bytes,
                    "sample_uris": stats.sample_uris,
                    "parser_results": stats.parser_results,
                }
                for ext, stats in self.by_extension.items()
            },
            "content_characteristics": {
                "languages": self.content_characteristics.languages,
                "document_types": self.content_characteristics.document_types,
                "quality_issues": self.content_characteristics.quality_issues,
            },
            "parser_recommendations": [
                {
                    "extension": r.extension,
                    "parser_id": r.parser_id,
                    "confidence": r.confidence,
                    "notes": r.notes,
                }
                for r in self.parser_recommendations
            ],
            "uncertainties": [
                {
                    "severity": u.severity,
                    "message": u.message,
                    "context": u.context,
                }
                for u in self.uncertainties
            ],
            "summary": self.summary,
        }


class SourceAnalyzer(SubAgent):
    """Sub-agent that investigates data sources.

    The SourceAnalyzer systematically analyzes a data source to understand
    its composition and recommend appropriate processing strategies. It
    uses specialized tools to enumerate, sample, and test-parse files.

    Context Requirements:
        - source_id: UUID of the source to analyze
        - connector: Connector instance for the source
        - user_intent: Optional user description of their goal

    Returns:
        SourceAnalysis with:
        - File composition statistics
        - Content characteristics (languages, types, quality)
        - Parser recommendations
        - Uncertainties for user review
    """

    AGENT_ID: ClassVar[str] = "source_analyzer"
    MAX_TURNS: ClassVar[int] = 30  # May need many tool calls for large sources
    TIMEOUT_SECONDS: ClassVar[int] = 300  # 5 minutes

    SYSTEM_PROMPT: ClassVar[
        str
    ] = """You are a source analysis agent for Semantik, a semantic search engine.

Your job: Investigate a data source and produce a comprehensive analysis that helps
the pipeline builder choose the right parsing, chunking, and embedding strategy.

You have tools to enumerate files, sample by type, and try parsers. Use them
systematically:

1. First, enumerate to understand the source composition (file types, sizes, counts)
2. Sample representative files from each major type
3. Try parsing samples to verify they work and understand content characteristics
4. Look for patterns: Are PDFs scanned? Is there mixed language content? Code vs prose?
5. Note any issues or uncertainties

Be thorough but efficient. Don't try every file - sample intelligently.

When done, produce a structured analysis with:
- Source composition (counts by type, size distribution)
- Content characteristics (languages, document types, quality issues)
- Recommended parsers for each file type with confidence
- Any uncertainties the user should know about

Always end with a structured JSON result in your final message using this format:

```json
{
  "total_files": 247,
  "total_size_bytes": 125000000,
  "by_extension": {
    ".pdf": {"count": 200, "total_size_bytes": 100000000},
    ".md": {"count": 47, "total_size_bytes": 25000000}
  },
  "content_characteristics": {
    "languages": ["en"],
    "document_types": ["academic", "notes"],
    "quality_issues": ["some_scanned_pdfs"]
  },
  "parser_recommendations": [
    {"extension": ".pdf", "parser_id": "unstructured", "confidence": 0.9, "notes": "Good for complex layouts"},
    {"extension": ".md", "parser_id": "text", "confidence": 1.0, "notes": null}
  ],
  "uncertainties": [
    {"severity": "notable", "message": "5 PDFs appear to be scanned documents"}
  ],
  "summary": "247 files, mostly academic PDFs with some markdown notes. Good quality overall."
}
```"""

    TOOLS: ClassVar[list[type[BaseTool]]] = [
        EnumerateFilesTool,
        SampleFilesTool,
        TryParserTool,
        DetectLanguageTool,
        GetFileContentPreviewTool,
    ]

    def _build_initial_message(self) -> Message:
        """Build the initial message with source context."""
        source_id = self.context.get("source_id", "unknown")
        user_intent = self.context.get("user_intent", "")

        content = f"""Analyze source {source_id} for pipeline configuration.

User's goal: {user_intent or "Not specified - infer from content"}

Produce a comprehensive analysis I can use to recommend a pipeline.

Start by enumerating the files to understand the source composition."""

        return Message(role="user", content=content)

    def _extract_result(self, response: Message) -> SubAgentResult:
        """Extract structured SourceAnalysis from the agent's final response."""
        try:
            # Parse the JSON from the response
            analysis_data = self._parse_analysis_json(response.content)

            if not analysis_data:
                # If no JSON found, create a basic result from what we know
                return SubAgentResult(
                    success=False,
                    data={},
                    summary="Could not extract structured analysis from response",
                )

            # Convert to Uncertainty objects
            uncertainties = []
            for u in analysis_data.get("uncertainties", []):
                uncertainties.append(
                    Uncertainty(
                        severity=u.get("severity", "info"),
                        message=u.get("message", ""),
                        context=u.get("context"),
                    )
                )

            return SubAgentResult(
                success=True,
                data=analysis_data,
                uncertainties=uncertainties,
                summary=analysis_data.get("summary", "Analysis complete"),
            )

        except Exception as e:
            logger.error(f"Failed to extract result: {e}", exc_info=True)
            return SubAgentResult(
                success=False,
                data=self._get_partial_result(),
                summary=f"Failed to extract result: {e}",
            )

    def _parse_analysis_json(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from the agent's response.

        Handles both raw JSON and JSON embedded in markdown code blocks.
        """
        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse the whole content as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object anywhere in the content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _get_partial_result(self) -> dict[str, Any]:
        """Get partial result from tool executions so far."""
        # Check for enumeration results in context
        enumerated = self.context.get("_enumerated_files", [])
        if enumerated:
            # Build partial analysis from enumeration
            by_extension: dict[str, int] = {}
            total_size = 0

            for ref in enumerated:
                ext = ref.extension.lower() if ref.extension else "(no ext)"
                by_extension[ext] = by_extension.get(ext, 0) + 1
                total_size += ref.size_bytes

            return {
                "partial": True,
                "total_files": len(enumerated),
                "total_size_bytes": total_size,
                "by_extension": by_extension,
            }

        return {"partial": True, "total_files": 0}
