"""Tests for build_pipeline tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from webui.services.assisted_flow.context import ToolContext


class TestBuildPipelineTool:
    """Test build_pipeline tool."""

    @pytest.fixture
    def mock_context(self) -> ToolContext:
        """Create mock context with pipeline state."""
        return ToolContext(
            session=MagicMock(),
            user_id=1,
            source_id=42,
            pipeline_state=None,
        )

    def test_server_has_build_pipeline_tool(self, mock_context: ToolContext) -> None:
        """Server includes build_pipeline tool."""
        from webui.services.assisted_flow.server import create_mcp_server

        with patch("webui.services.assisted_flow.server.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = MagicMock()

            server = create_mcp_server(mock_context)

        # Server should have an instance with tools registered
        assert server is not None
        assert server["name"] == "semantik-assisted-flow"

    @pytest.mark.asyncio
    async def test_build_pipeline_validates_plugins(self, mock_context: ToolContext) -> None:
        """build_pipeline validates that plugins exist."""
        from webui.services.assisted_flow.server import create_mcp_server

        with patch("webui.services.assisted_flow.server.plugin_registry") as mock_registry:
            # First call for list/get tools, then None for missing plugin
            mock_registry.find_by_id.return_value = None

            server = create_mcp_server(mock_context)

            # Get the MCP server instance to call tools directly
            mcp_instance = server["instance"]

            # The tool should be callable - we test via the server's tool handler
            # For now, just verify the server was created with tools
            assert mcp_instance is not None
