"""Agent tools for pipeline template discovery.

These tools allow the agent to explore available pre-configured pipeline templates
that can be used as starting points for collection configuration.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from webui.services.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ListTemplatesTool(BaseTool):
    """List available pipeline templates.

    Returns a summary of each template including ID, name, description,
    suggested use cases, and tunable parameters. Templates provide
    pre-configured pipelines for common document types.
    """

    NAME: ClassVar[str] = "list_templates"
    DESCRIPTION: ClassVar[str] = (
        "List available pipeline templates. Templates are pre-configured pipelines "
        "for common use cases like academic papers, codebases, documentation, etc. "
        "Returns template IDs, names, descriptions, suggested use cases, and tunable parameters."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "suggested_for": {
                "type": "string",
                "description": (
                    "Optional keyword to filter templates by suggested use case " "(e.g., 'PDF', 'code', 'email')"
                ),
            },
        },
        "required": [],
    }

    async def execute(self, suggested_for: str | None = None) -> dict[str, Any]:
        """Execute the template listing.

        Args:
            suggested_for: Optional keyword to filter by suggested use case

        Returns:
            Dictionary with templates list and metadata
        """
        try:
            from shared.pipeline.templates import list_templates

            all_templates = list_templates()

            templates = []
            for template in all_templates:
                # Filter by suggested_for if provided
                if suggested_for:
                    keyword_lower = suggested_for.lower()
                    matches = any(keyword_lower in s.lower() for s in template.suggested_for)
                    if not matches:
                        continue

                template_info = {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "suggested_for": template.suggested_for,
                    "tunable_parameters": [
                        {
                            "path": t.path,
                            "description": t.description,
                            "default": t.default,
                            "range": list(t.range) if t.range else None,
                            "options": t.options,
                        }
                        for t in template.tunable
                    ],
                    "node_count": len(template.pipeline.nodes),
                    "edge_count": len(template.pipeline.edges),
                }
                templates.append(template_info)

            return {
                "templates": templates,
                "count": len(templates),
                "filter": suggested_for,
            }

        except ValueError as e:
            # Template validation errors
            logger.error(f"Template validation error: {e}", exc_info=True)
            return {
                "error": f"Template validation failed: {e}",
                "templates": [],
                "count": 0,
            }
        except Exception as e:
            logger.error(f"Failed to list templates: {e}", exc_info=True)
            return {
                "error": str(e),
                "templates": [],
                "count": 0,
            }


class GetTemplateDetailsTool(BaseTool):
    """Get detailed information about a specific pipeline template.

    Returns the full template including the complete pipeline DAG structure,
    all tunable parameters with their current defaults, and usage hints.
    """

    NAME: ClassVar[str] = "get_template_details"
    DESCRIPTION: ClassVar[str] = (
        "Get detailed information about a specific pipeline template. "
        "Returns the full pipeline DAG structure, node configurations, "
        "edge routing rules, and all tunable parameters."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "template_id": {
                "type": "string",
                "description": "The unique ID of the template to get details for",
            },
        },
        "required": ["template_id"],
    }

    async def execute(self, template_id: str) -> dict[str, Any]:
        """Execute the template details lookup.

        Args:
            template_id: The ID of the template to look up

        Returns:
            Dictionary with template details or error
        """
        try:
            from shared.pipeline.templates import list_templates, load_template

            template = load_template(template_id)

            if not template:
                # List available templates to help the agent
                available = [t.id for t in list_templates()]
                return {
                    "found": False,
                    "error": f"Template '{template_id}' not found",
                    "available_templates": available,
                }

            # Build detailed response with full pipeline structure
            details: dict[str, Any] = {
                "found": True,
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "suggested_for": template.suggested_for,
                "pipeline": template.pipeline.to_dict(),
                "tunable_parameters": [
                    {
                        "path": t.path,
                        "description": t.description,
                        "default": t.default,
                        "range": list(t.range) if t.range else None,
                        "options": t.options,
                    }
                    for t in template.tunable
                ],
                "nodes": [
                    {
                        "id": node.id,
                        "type": node.type.value,
                        "plugin_id": node.plugin_id,
                        "config": node.config,
                    }
                    for node in template.pipeline.nodes
                ],
                "edges": [
                    {
                        "from": edge.from_node,
                        "to": edge.to_node,
                        "when": edge.when,
                    }
                    for edge in template.pipeline.edges
                ],
            }

            return details

        except ValueError as e:
            logger.error(f"Template validation error: {e}", exc_info=True)
            return {
                "found": False,
                "error": f"Template validation failed: {e}",
            }
        except Exception as e:
            logger.error(f"Failed to get template details for '{template_id}': {e}", exc_info=True)
            return {
                "found": False,
                "error": str(e),
            }
