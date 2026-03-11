"""Tests for assisted flow prompts."""


class TestBuildInitialPrompt:
    """Test build_initial_prompt function."""

    def test_includes_source_stats(self) -> None:
        """Prompt includes source statistics."""
        from webui.services.assisted_flow.prompts import build_initial_prompt

        stats = {
            "source_name": "My Docs",
            "source_type": "directory",
            "source_path": "/data/docs",
            "source_config": {"path": "/data/docs"},
        }

        prompt = build_initial_prompt(stats)

        assert "My Docs" in prompt
        assert "directory" in prompt
        assert "/data/docs" in prompt

    def test_includes_user_message(self) -> None:
        """Prompt includes placeholder for user intent."""
        from webui.services.assisted_flow.prompts import build_initial_prompt

        stats = {
            "source_name": "Test",
            "source_type": "directory",
            "source_path": "/test",
            "source_config": {},
        }

        prompt = build_initial_prompt(stats)

        assert "configure" in prompt.lower() or "pipeline" in prompt.lower()


class TestSystemPrompt:
    """Test SYSTEM_PROMPT constant."""

    def test_system_prompt_defined(self) -> None:
        """System prompt is defined and non-empty."""
        from webui.services.assisted_flow.prompts import SYSTEM_PROMPT

        assert SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 100
        assert "pipeline" in SYSTEM_PROMPT.lower()
