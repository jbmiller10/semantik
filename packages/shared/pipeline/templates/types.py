"""Type definitions for pipeline templates.

This module defines the core data structures for pre-configured pipeline templates
that can be used to bootstrap new collections with common use case configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.pipeline.types import PipelineDAG, PipelineNode


@dataclass(frozen=True)
class TunableParameter:
    """A parameter in a pipeline template that can be adjusted by users/agents.

    Tunable parameters allow templates to be customized without rebuilding
    the entire DAG. Each parameter has a path that resolves to a specific
    config value in a pipeline node.

    Attributes:
        path: Dot-notation path to the parameter (e.g., "nodes.chunker.config.max_tokens")
        description: Human-readable description of what this parameter controls
        default: The default value for this parameter
        range: Optional (min, max) tuple for numeric parameters
        options: Optional list of valid string options for enum-like parameters
    """

    path: str
    description: str
    default: Any
    range: tuple[int, int] | None = None
    options: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {
            "path": self.path,
            "description": self.description,
            "default": self.default,
        }
        if self.range is not None:
            result["range"] = list(self.range)
        if self.options is not None:
            result["options"] = list(self.options)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunableParameter:
        """Create a TunableParameter from a dictionary."""
        range_val = data.get("range")
        return cls(
            path=data["path"],
            description=data["description"],
            default=data["default"],
            range=tuple(range_val) if range_val else None,
            options=data.get("options"),
        )


@dataclass(frozen=True)
class PipelineTemplate:
    """A pre-configured pipeline template for common use cases.

    Templates provide ready-to-use pipeline configurations for specific
    document types or workflows. Each template includes a complete PipelineDAG
    and optional tunable parameters for customization.

    Attributes:
        id: Unique identifier for this template (e.g., "academic-papers")
        name: Human-readable name (e.g., "Academic Papers")
        description: Detailed description of what this template is designed for
        suggested_for: List of use case hints (e.g., ["PDF", "research", "citations"])
        pipeline: The pre-configured PipelineDAG definition
        tunable: List of parameters that can be adjusted after template selection
    """

    id: str
    name: str
    description: str
    suggested_for: list[str]
    pipeline: PipelineDAG
    tunable: list[TunableParameter] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "suggested_for": list(self.suggested_for),
            "pipeline": self.pipeline.to_dict(),
            "tunable": [t.to_dict() for t in self.tunable],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineTemplate:
        """Create a PipelineTemplate from a dictionary."""
        from shared.pipeline.types import PipelineDAG

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            suggested_for=data["suggested_for"],
            pipeline=PipelineDAG.from_dict(data["pipeline"]),
            tunable=[TunableParameter.from_dict(t) for t in data.get("tunable", [])],
        )


def resolve_tunable_path(template: PipelineTemplate, path: str) -> tuple[PipelineNode | None, str | None]:
    """Resolve a tunable parameter path to its target node and config key.

    Paths follow the format: "nodes.<node_id>.config.<param_name>"

    Args:
        template: The pipeline template containing the DAG
        path: The dot-notation path to resolve

    Returns:
        Tuple of (PipelineNode, config_key) if path is valid,
        (None, None) if the path cannot be resolved.

    Example:
        >>> node, key = resolve_tunable_path(template, "nodes.chunker.config.max_tokens")
        >>> if node:
        ...     print(f"Resolved to node {node.id}, config key {key}")
    """
    parts = path.split(".")

    # Expected format: nodes.<node_id>.config.<param_name>
    if len(parts) != 4:
        return (None, None)

    if parts[0] != "nodes" or parts[2] != "config":
        return (None, None)

    node_id = parts[1]
    config_key = parts[3]

    # Find the node in the DAG
    for node in template.pipeline.nodes:
        if node.id == node_id:
            return (node, config_key)

    return (None, None)


__all__ = [
    "TunableParameter",
    "PipelineTemplate",
    "resolve_tunable_path",
]
