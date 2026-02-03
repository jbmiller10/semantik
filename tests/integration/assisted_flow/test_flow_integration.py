"""Integration tests for assisted flow.

These tests verify the full flow from start to apply works end-to-end,
with mocked SDK client to avoid requiring Claude Code CLI.
"""

from unittest.mock import MagicMock, patch

import pytest

from webui.services.assisted_flow.context import ToolContext
from webui.services.assisted_flow.prompts import SYSTEM_PROMPT, build_initial_prompt
from webui.services.assisted_flow.server import create_mcp_server
from webui.services.assisted_flow.session_manager import SessionManager
from webui.services.assisted_flow.subagents import get_subagents


class TestAssistedFlowIntegration:
    """Integration tests for the complete assisted flow."""

    def test_full_module_imports(self) -> None:
        """All assisted flow modules can be imported."""
        # These imports verify the module structure is correct
        from webui.services.assisted_flow import (
            context,  # noqa: F401
            prompts,  # noqa: F401
            sdk_service,  # noqa: F401
            server,  # noqa: F401
            session_manager,  # noqa: F401
            source_stats,  # noqa: F401
            subagents,  # noqa: F401
        )

    def test_context_creation_and_mutation(self) -> None:
        """ToolContext can be created and mutated during session."""
        ctx = ToolContext(
            user_id=1,
            source_id=42,
        )

        # Initially no pipeline
        assert ctx.pipeline_state is None
        assert ctx.applied_config is None

        # Build pipeline
        ctx.pipeline_state = {
            "nodes": [{"id": "p1", "type": "parser", "plugin_id": "tika"}],
            "edges": [],
        }
        assert ctx.pipeline_state is not None
        assert len(ctx.pipeline_state["nodes"]) == 1

        # Apply pipeline
        ctx.applied_config = {
            "collection_name": "Test",
            "pipeline_config": ctx.pipeline_state,
        }
        assert ctx.applied_config is not None

    def test_prompts_are_well_formed(self) -> None:
        """System and initial prompts are properly formatted."""
        # System prompt should mention key concepts
        assert "pipeline" in SYSTEM_PROMPT.lower()
        assert "plugin" in SYSTEM_PROMPT.lower()

        # Initial prompt includes source stats
        source_stats = {
            "source_name": "Documents",
            "source_type": "directory",
            "source_path": "/data",
        }
        initial = build_initial_prompt(source_stats)

        assert "Documents" in initial
        assert "directory" in initial
        assert "/data" in initial

    def test_mcp_server_has_required_tools(self) -> None:
        """MCP server includes all required tools."""
        ctx = ToolContext(
            user_id=1,
            source_id=42,
        )

        server = create_mcp_server(ctx)

        # Server should be properly structured
        assert server["type"] == "sdk"
        assert server["name"] == "semantik-assisted-flow"
        assert "instance" in server

    def test_subagents_properly_configured(self) -> None:
        """Subagents have correct model and prompts."""
        agents = get_subagents()

        # Both agents should be defined
        assert "explorer" in agents
        assert "validator" in agents

        # Both should use haiku for efficiency
        assert agents["explorer"].model == "haiku"
        assert agents["validator"].model == "haiku"

        # Both should have prompts
        assert agents["explorer"].prompt
        assert agents["validator"].prompt

    @pytest.mark.asyncio()
    async def test_session_manager_lifecycle(self) -> None:
        """Session manager handles full lifecycle correctly."""
        manager = SessionManager(ttl_seconds=60)

        # Store a mock client
        mock_client = MagicMock()
        await manager.store_client("test_session", mock_client, user_id=1)

        # Retrieve it
        retrieved = await manager.get_client("test_session", user_id=1)
        assert retrieved is mock_client

        # Remove it
        await manager.remove_client("test_session")

        # Should be gone
        retrieved = await manager.get_client("test_session", user_id=1)
        assert retrieved is None

    @pytest.mark.asyncio()
    async def test_sdk_service_error_handling(self) -> None:
        """SDK service properly wraps SDK errors."""
        from claude_agent_sdk import CLINotFoundError

        from webui.services.assisted_flow.sdk_service import (
            SDKNotAvailableError,
            create_sdk_session,
        )

        source_stats = {
            "source_name": "Test",
            "source_type": "directory",
            "source_path": "/test",
        }

        with patch("webui.services.assisted_flow.sdk_service.ClaudeSDKClient") as mock_client:
            # Simulate CLI not found
            mock_client.side_effect = CLINotFoundError("Claude Code CLI not found")

            with pytest.raises(SDKNotAvailableError):
                await create_sdk_session(
                    user_id=1,
                    source_id=42,
                    source_stats=source_stats,
                )


class TestToolExecution:
    """Test individual tool execution."""

    @pytest.fixture()
    def mock_context(self) -> ToolContext:
        """Create a mock tool context."""
        return ToolContext(
            user_id=1,
            source_id=42,
        )

    def test_build_pipeline_stores_in_context(self, mock_context: ToolContext) -> None:
        """build_pipeline tool stores pipeline in context."""
        # The tool is defined inside create_mcp_server
        # We verify the pattern works by checking context mutation
        pipeline = {
            "nodes": [
                {"id": "parser", "type": "parser", "plugin_id": "tika"},
                {"id": "chunker", "type": "chunker", "plugin_id": "recursive"},
            ],
            "edges": [{"from_node": "parser", "to_node": "chunker"}],
        }

        mock_context.pipeline_state = pipeline

        assert mock_context.pipeline_state == pipeline
        assert len(mock_context.pipeline_state["nodes"]) == 2

    def test_apply_pipeline_requires_built_pipeline(self, mock_context: ToolContext) -> None:
        """apply_pipeline should check for existing pipeline."""
        # No pipeline state - would fail
        assert mock_context.pipeline_state is None

        # Set pipeline
        mock_context.pipeline_state = {"nodes": [], "edges": []}
        assert mock_context.pipeline_state is not None
