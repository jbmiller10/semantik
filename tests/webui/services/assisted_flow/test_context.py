"""Tests for assisted flow tool context."""

from webui.services.assisted_flow.context import ToolContext


class TestToolContext:
    """Test ToolContext dataclass."""

    def test_create_context(self) -> None:
        """Context stores user_id and source_id."""
        user_id = 123
        source_id = 456

        ctx = ToolContext(
            user_id=user_id,
            source_id=source_id,
        )

        assert ctx.user_id == user_id
        assert ctx.source_id == source_id
        assert ctx.pipeline_state is None

    def test_context_with_pipeline_state(self) -> None:
        """Context can hold current pipeline state."""
        pipeline = {"nodes": [], "edges": []}

        ctx = ToolContext(
            user_id=1,
            source_id=42,
            pipeline_state=pipeline,
        )

        assert ctx.pipeline_state == pipeline
