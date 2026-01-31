"""PipelineValidator sub-agent for validating pipeline configurations.

The PipelineValidator is a specialized sub-agent that validates a pipeline
configuration against sample files from a data source. It:

1. Runs dry-run validation on sample files
2. Investigates failures to understand their causes
3. Categorizes failures (parser issue, encoding, corruption, etc.)
4. Tries alternative configurations for fixable issues
5. Produces a structured validation report with recommendations

Returns a structured ValidationReport with assessment and suggestions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from webui.services.agent.subagents.base import Message, SubAgent, SubAgentResult, Uncertainty
from webui.services.agent.tools.subagent_tools.validation import (
    CompareParserOutputTool,
    GetFailureDetailsTool,
    InspectChunksTool,
    RunDryRunTool,
    TryAlternativeConfigTool,
)

if TYPE_CHECKING:
    from webui.services.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class FailureCategory:
    """A category of failure with affected files.

    Attributes:
        category: The failure category (parser_error, encoding_error, etc.)
        count: Number of files in this category
        example_files: Sample URIs for reference
        is_fixable: Whether this category can be fixed with config changes
        suggested_fix: Suggested fix if fixable
    """

    category: str
    count: int
    example_files: list[str] = field(default_factory=list)
    is_fixable: bool = False
    suggested_fix: str | None = None


@dataclass
class PipelineFix:
    """A suggested fix for a pipeline issue.

    Attributes:
        issue: Description of the issue
        fix_type: Type of fix (parser_change, config_change, filter_files)
        details: Fix implementation details
        affected_files: How many files this would help
        confidence: How confident we are in this fix (0.0-1.0)
    """

    issue: str
    fix_type: Literal["parser_change", "config_change", "filter_files", "accept"]
    details: dict[str, Any]
    affected_files: int
    confidence: float


@dataclass
class ValidationReport:
    """Complete validation report for a pipeline configuration.

    Attributes:
        success_rate: Percentage of files that passed (0.0-1.0)
        files_tested: Total number of files tested
        files_passed: Number of files that succeeded
        files_failed: Number of files that failed
        assessment: Overall assessment (ready, needs_review, blocking_issues)
        failure_categories: Breakdown of failures by category
        suggested_fixes: Recommended fixes for issues
        chunk_quality: Assessment of chunking quality
        summary: Human-readable summary
    """

    success_rate: float
    files_tested: int
    files_passed: int
    files_failed: int
    assessment: Literal["ready", "needs_review", "blocking_issues"]
    failure_categories: list[FailureCategory] = field(default_factory=list)
    suggested_fixes: list[PipelineFix] = field(default_factory=list)
    chunk_quality: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success_rate": self.success_rate,
            "files_tested": self.files_tested,
            "files_passed": self.files_passed,
            "files_failed": self.files_failed,
            "assessment": self.assessment,
            "failure_categories": [
                {
                    "category": c.category,
                    "count": c.count,
                    "example_files": c.example_files,
                    "is_fixable": c.is_fixable,
                    "suggested_fix": c.suggested_fix,
                }
                for c in self.failure_categories
            ],
            "suggested_fixes": [
                {
                    "issue": f.issue,
                    "fix_type": f.fix_type,
                    "details": f.details,
                    "affected_files": f.affected_files,
                    "confidence": f.confidence,
                }
                for f in self.suggested_fixes
            ],
            "chunk_quality": self.chunk_quality,
            "summary": self.summary,
        }


class PipelineValidator(SubAgent):
    """Sub-agent that validates pipeline configurations.

    The PipelineValidator systematically tests a pipeline configuration
    against sample files, investigates failures, and produces a structured
    validation report with recommendations.

    Context Requirements:
        - pipeline: dict (the pipeline DAG to validate)
        - sample_files: list[FileReference] (files to test)
        - connector: Connector instance
        - session: Database session

    Returns:
        ValidationReport with:
        - Success rate and file counts
        - Failure categorization
        - Suggested fixes
        - Chunk quality assessment
        - Overall assessment and summary
    """

    AGENT_ID: ClassVar[str] = "pipeline_validator"
    MAX_TURNS: ClassVar[int] = 25
    TIMEOUT_SECONDS: ClassVar[int] = 300  # 5 minutes

    SYSTEM_PROMPT: ClassVar[
        str
    ] = """You are a pipeline validation agent for Semantik, a semantic search engine.

Your job: Test a pipeline configuration and produce a validation report that helps
the user understand if their pipeline is ready for production.

You have tools to run dry-run validation, investigate failures, try alternatives,
and inspect chunk output. Use them systematically:

1. First, run dry-run validation on the sample files to get overall success rate
2. If there are failures, investigate each category:
   - Use get_failure_details for specific errors
   - For parser errors, try try_alternative_config with different parsers
   - For encoding errors, check if it's fixable with config changes
3. Inspect chunk output for successful files to verify quality
4. Produce your assessment:
   - >95% success rate: "ready" - pipeline is production-ready
   - 90-95% success rate: "needs_review" - notable issues to consider
   - <90% success rate: "blocking_issues" - must fix before production

Be thorough but efficient. Focus on understanding failure patterns and finding fixes.

When done, produce a structured JSON result in your final message using this format:

```json
{
  "success_rate": 0.96,
  "files_tested": 50,
  "files_passed": 48,
  "files_failed": 2,
  "assessment": "ready",
  "failure_categories": [
    {"category": "encoding_error", "count": 2, "example_files": ["file1.pdf"], "is_fixable": false}
  ],
  "suggested_fixes": [
    {"issue": "2 files have encoding issues", "fix_type": "accept", "details": {}, "affected_files": 2, "confidence": 0.9}
  ],
  "chunk_quality": {
    "avg_chunk_size": 450,
    "size_variance": "normal",
    "assessment": "good"
  },
  "summary": "Pipeline is ready. 96% success rate with 2 files having encoding issues that can be safely skipped."
}
```"""

    TOOLS: ClassVar[list[type[BaseTool]]] = [
        RunDryRunTool,
        GetFailureDetailsTool,
        TryAlternativeConfigTool,
        CompareParserOutputTool,
        InspectChunksTool,
    ]

    def _build_initial_message(self) -> Message:
        """Build the initial message with pipeline context."""
        pipeline = self.context.get("pipeline", {})
        sample_files = self.context.get("sample_files", [])

        # Get pipeline ID and summary
        pipeline_id = pipeline.get("id", "unknown")
        node_count = len(pipeline.get("nodes", []))
        sample_count = len(sample_files)

        # List sample file extensions for context
        extensions: dict[str, int] = {}
        for f in sample_files[:100]:  # Limit for initial message
            ext = getattr(f, "extension", None) or "(no ext)"
            extensions[ext] = extensions.get(ext, 0) + 1

        ext_summary = ", ".join(f"{ext}: {count}" for ext, count in sorted(extensions.items(), key=lambda x: -x[1])[:5])

        content = f"""Validate pipeline "{pipeline_id}" with {node_count} nodes against {sample_count} sample files.

File composition: {ext_summary}

Produce a validation report assessing:
1. Overall success rate
2. Failure categories and their fixability
3. Chunk quality for successful files
4. Recommended fixes if any
5. Final assessment (ready / needs_review / blocking_issues)

Start by running dry-run validation to see the overall success rate."""

        return Message(role="user", content=content)

    def _extract_result(self, response: Message) -> SubAgentResult:
        """Extract structured ValidationReport from the agent's final response."""
        try:
            # Parse the JSON from the response
            report_data = self._parse_report_json(response.content)

            if not report_data:
                # If no JSON found, create a basic result
                return SubAgentResult(
                    success=False,
                    data={},
                    summary="Could not extract structured validation report from response",
                )

            # Create uncertainties based on assessment
            uncertainties = []
            success_rate = report_data.get("success_rate", 0)
            assessment = report_data.get("assessment", "blocking_issues")

            if assessment == "blocking_issues":
                uncertainties.append(
                    Uncertainty(
                        severity="blocking",
                        message=f"Pipeline validation failed with {success_rate:.0%} success rate",
                        context={
                            "files_tested": report_data.get("files_tested"),
                            "files_failed": report_data.get("files_failed"),
                        },
                    )
                )
            elif assessment == "needs_review":
                uncertainties.append(
                    Uncertainty(
                        severity="notable",
                        message=f"Pipeline has notable issues ({success_rate:.0%} success rate)",
                        context={
                            "failure_categories": [
                                c.get("category") for c in report_data.get("failure_categories", [])
                            ],
                        },
                    )
                )

            # Add uncertainties for non-fixable failures
            for category in report_data.get("failure_categories", []):
                if not category.get("is_fixable", False) and category.get("count", 0) > 0:
                    uncertainties.append(
                        Uncertainty(
                            severity="info",
                            message=f"{category.get('count')} files failed with {category.get('category')}",
                            context={"example_files": category.get("example_files", [])[:3]},
                        )
                    )

            return SubAgentResult(
                success=True,
                data=report_data,
                uncertainties=uncertainties,
                summary=report_data.get("summary", "Validation complete"),
            )

        except Exception as e:
            logger.error(f"Failed to extract validation result: {e}", exc_info=True)
            return SubAgentResult(
                success=False,
                data=self._get_partial_result(),
                summary=f"Failed to extract result: {e}",
            )

    def _parse_report_json(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from the agent's response.

        Handles both raw JSON and JSON embedded in markdown code blocks.
        """
        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            try:
                result: dict[str, Any] = json.loads(json_match.group(1))
                return result
            except json.JSONDecodeError:
                pass

        # Try to parse the whole content as JSON
        try:
            result: dict[str, Any] = json.loads(content)
            return result
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object anywhere in the content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except json.JSONDecodeError:
                pass

        return None

    def _get_partial_result(self) -> dict[str, Any]:
        """Get partial result from dry-run if available."""
        # Check for dry-run results in context
        dry_run_result = self.context.get("_dry_run_result")
        if dry_run_result:
            total = dry_run_result.files_succeeded + dry_run_result.files_failed
            success_rate = dry_run_result.files_succeeded / total if total > 0 else 0.0

            return {
                "partial": True,
                "success_rate": success_rate,
                "files_tested": total,
                "files_passed": dry_run_result.files_succeeded,
                "files_failed": dry_run_result.files_failed,
                "assessment": (
                    "ready" if success_rate >= 0.95 else "needs_review" if success_rate >= 0.90 else "blocking_issues"
                ),
            }

        return {"partial": True, "success_rate": 0, "files_tested": 0}


__all__ = [
    "PipelineValidator",
    "ValidationReport",
    "FailureCategory",
    "PipelineFix",
]
