"""Tool implementations for the agent service.

Tools provide the capabilities that agents use to interact with the system.
The orchestrator and sub-agents each have their own specialized toolsets.
"""

from webui.services.agent.tools.base import BaseTool
from webui.services.agent.tools.pipeline import (
    ApplyPipelineTool,
    BuildPipelineTool,
    GetPipelineStateTool,
)
from webui.services.agent.tools.plugins import GetPluginDetailsTool, ListPluginsTool
from webui.services.agent.tools.spawn import SpawnSourceAnalyzerTool
from webui.services.agent.tools.templates import GetTemplateDetailsTool, ListTemplatesTool

__all__ = [
    # Base
    "BaseTool",
    # Plugin tools
    "ListPluginsTool",
    "GetPluginDetailsTool",
    # Template tools
    "ListTemplatesTool",
    "GetTemplateDetailsTool",
    # Pipeline tools
    "GetPipelineStateTool",
    "BuildPipelineTool",
    "ApplyPipelineTool",
    # Spawn tools
    "SpawnSourceAnalyzerTool",
]
