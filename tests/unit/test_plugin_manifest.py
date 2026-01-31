"""Unit tests for plugin manifest serialization."""

from shared.plugins.manifest import AgentHints, PluginManifest


class TestAgentHints:
    """Tests for AgentHints dataclass."""

    def test_to_dict_required_fields_only(self):
        """Test serialization with only required fields."""
        hints = AgentHints(
            purpose="Test purpose",
            best_for=["scenario1", "scenario2"],
            not_recommended_for=["bad1"],
        )
        result = hints.to_dict()
        assert result == {
            "purpose": "Test purpose",
            "best_for": ["scenario1", "scenario2"],
            "not_recommended_for": ["bad1"],
        }

    def test_to_dict_all_fields(self):
        """Test serialization with all fields populated."""
        hints = AgentHints(
            purpose="Full test",
            best_for=["good"],
            not_recommended_for=["bad"],
            input_types=["text/plain", "application/pdf"],
            output_type="text",
            tradeoffs="Speed vs accuracy",
            examples=[{"name": "example1", "config": {"key": "value"}}],
        )
        result = hints.to_dict()
        assert result["input_types"] == ["text/plain", "application/pdf"]
        assert result["output_type"] == "text"
        assert result["tradeoffs"] == "Speed vs accuracy"
        assert result["examples"] == [{"name": "example1", "config": {"key": "value"}}]

    def test_from_dict_required_fields_only(self):
        """Test deserialization with only required fields."""
        data = {
            "purpose": "Test",
            "best_for": ["a"],
            "not_recommended_for": ["b"],
        }
        hints = AgentHints.from_dict(data)
        assert hints.purpose == "Test"
        assert hints.best_for == ["a"]
        assert hints.not_recommended_for == ["b"]
        assert hints.input_types is None
        assert hints.output_type is None

    def test_round_trip(self):
        """Test serialization round-trip preserves data."""
        original = AgentHints(
            purpose="Round trip test",
            best_for=["x", "y"],
            not_recommended_for=["z"],
            input_types=["text/*"],
            output_type="vectors",
            tradeoffs="Some tradeoff",
            examples=[{"name": "ex", "config": {}}],
        )
        restored = AgentHints.from_dict(original.to_dict())
        assert restored == original


class TestPluginManifest:
    """Tests for PluginManifest with agent_hints."""

    def test_to_dict_without_agent_hints(self):
        """Test backward compatibility - no agent_hints field."""
        manifest = PluginManifest(
            id="test",
            type="embedding",
            version="1.0.0",
            display_name="Test Plugin",
            description="A test plugin",
        )
        result = manifest.to_dict()
        assert "agent_hints" not in result

    def test_to_dict_with_agent_hints(self):
        """Test serialization includes agent_hints when present."""
        manifest = PluginManifest(
            id="test",
            type="parser",
            version="1.0.0",
            display_name="Test Parser",
            description="Parses files",
            agent_hints=AgentHints(
                purpose="Parse documents",
                best_for=["PDFs"],
                not_recommended_for=["code"],
            ),
        )
        result = manifest.to_dict()
        assert "agent_hints" in result
        assert result["agent_hints"]["purpose"] == "Parse documents"

    def test_from_dict_without_agent_hints(self):
        """Test deserialization without agent_hints (backward compat)."""
        data = {
            "id": "test",
            "type": "embedding",
            "version": "1.0.0",
            "display_name": "Test",
            "description": "Test desc",
        }
        manifest = PluginManifest.from_dict(data)
        assert manifest.id == "test"
        assert manifest.agent_hints is None

    def test_from_dict_with_agent_hints(self):
        """Test deserialization includes agent_hints."""
        data = {
            "id": "test",
            "type": "parser",
            "version": "1.0.0",
            "display_name": "Test",
            "description": "Test desc",
            "agent_hints": {
                "purpose": "Parse things",
                "best_for": ["docs"],
                "not_recommended_for": ["code"],
            },
        }
        manifest = PluginManifest.from_dict(data)
        assert manifest.agent_hints is not None
        assert manifest.agent_hints.purpose == "Parse things"

    def test_round_trip(self):
        """Test full manifest round-trip."""
        original = PluginManifest(
            id="full-test",
            type="parser",
            version="2.0.0",
            display_name="Full Test",
            description="Complete test",
            author="Test Author",
            agent_hints=AgentHints(
                purpose="Full round trip",
                best_for=["everything"],
                not_recommended_for=["nothing"],
                output_type="text",
            ),
        )
        restored = PluginManifest.from_dict(original.to_dict())
        assert restored == original
