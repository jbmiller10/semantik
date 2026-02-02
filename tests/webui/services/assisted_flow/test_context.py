"""Tests for assisted flow tool context."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from webui.services.assisted_flow.context import ToolContext


class TestToolContext:
    """Test ToolContext dataclass."""

    def test_create_context(self) -> None:
        """Context stores session, user_id, source_id."""
        session = MagicMock()
        user_id = 123
        source_id = uuid4()

        ctx = ToolContext(
            session=session,
            user_id=user_id,
            source_id=source_id,
        )

        assert ctx.session is session
        assert ctx.user_id == user_id
        assert ctx.source_id == source_id
        assert ctx.pipeline_state is None

    def test_context_with_pipeline_state(self) -> None:
        """Context can hold current pipeline state."""
        session = MagicMock()
        pipeline = {"nodes": [], "edges": []}

        ctx = ToolContext(
            session=session,
            user_id=1,
            source_id=uuid4(),
            pipeline_state=pipeline,
        )

        assert ctx.pipeline_state == pipeline
