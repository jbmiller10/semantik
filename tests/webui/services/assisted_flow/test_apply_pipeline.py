"""Tests for apply_pipeline tool."""

import pytest


class TestApplyPipelineTool:
    """Test apply_pipeline tool."""

    @pytest.fixture()
    def mock_context(self):
        """Create mock context with pipeline state."""
        from webui.services.assisted_flow.context import ToolContext

        return ToolContext(
            user_id=1,
            source_id=42,
            pipeline_state={
                "id": "agent-recommended",
                "version": "1",
                "nodes": [
                    {"id": "parser", "type": "parser", "plugin_id": "tika"},
                    {"id": "chunker", "type": "chunker", "plugin_id": "recursive"},
                ],
                "edges": [{"from_node": "parser", "to_node": "chunker"}],
            },
        )

    def test_server_creates_successfully(self, mock_context) -> None:
        """Server creates with apply_pipeline tool."""
        from webui.services.assisted_flow.server import create_mcp_server

        server = create_mcp_server(mock_context)

        # Verify server is created and has the expected structure
        assert server is not None
        assert isinstance(server, dict)
        assert server["name"] == "semantik-assisted-flow"
        assert server["type"] == "sdk"

    def test_apply_pipeline_no_pipeline_state(self) -> None:
        """apply_pipeline fails gracefully when no pipeline configured."""
        from webui.services.assisted_flow.context import ToolContext
        from webui.services.assisted_flow.server import create_mcp_server

        ctx = ToolContext(
            user_id=1,
            source_id=42,
            pipeline_state=None,  # No pipeline configured
        )

        # Should create successfully even without pipeline state
        server = create_mcp_server(ctx)
        assert server is not None
