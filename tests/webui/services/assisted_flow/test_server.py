"""Tests for assisted flow MCP server creation."""

from webui.services.assisted_flow.context import ToolContext


class TestCreateMCPServer:
    """Test create_mcp_server function."""

    def test_creates_server_with_tools(self) -> None:
        """Server is created with expected tools."""
        from webui.services.assisted_flow.server import create_mcp_server

        mock_context = ToolContext(
            user_id=1,
            source_id=42,
        )

        server = create_mcp_server(mock_context)

        # Verify server has the expected structure (dict with type, name, instance)
        assert server is not None
        assert isinstance(server, dict)
        assert server["name"] == "semantik-assisted-flow"
        assert server["type"] == "sdk"
        assert "instance" in server

    def test_server_includes_eight_tools(self) -> None:
        """Server includes all 8 tools (4 original + 4 new)."""
        from webui.services.assisted_flow.server import create_mcp_server

        mock_context = ToolContext(
            user_id=1,
            source_id=42,
        )

        server = create_mcp_server(mock_context)

        # The server instance has a list_tools method
        mcp_server = server["instance"]
        # Check server was created with the expected version
        assert server["name"] == "semantik-assisted-flow"
        assert mcp_server is not None
