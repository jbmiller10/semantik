"""Unit tests for plugin discovery tools."""

from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.manifest import AgentHints, PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource
from webui.services.agent.tools.plugins import GetPluginDetailsTool, ListPluginsTool


@pytest.fixture()
def sample_manifest():
    """Create a sample plugin manifest."""
    return PluginManifest(
        id="test-parser",
        type="parser",
        version="1.0.0",
        display_name="Test Parser",
        description="A test parser for unit testing",
        author="Test Author",
        license="MIT",
        homepage="https://example.com",
        requires=["python>=3.9"],
        capabilities={"supports_pdf": True, "supports_docx": True},
        agent_hints=AgentHints(
            purpose="Parse documents into text",
            best_for=["PDF files", "DOCX files"],
            not_recommended_for=["Binary files"],
            input_types=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            output_type="text",
            tradeoffs="Fast but may miss some formatting",
        ),
    )


@pytest.fixture()
def sample_manifest_no_hints():
    """Create a manifest without agent hints."""
    return PluginManifest(
        id="minimal-plugin",
        type="chunker",
        version="0.1.0",
        display_name="Minimal Plugin",
        description="A minimal plugin",
    )


@pytest.fixture()
def sample_record(sample_manifest):
    """Create a sample plugin record."""
    return PluginRecord(
        plugin_type="parser",
        plugin_id="test-parser",
        plugin_version="1.0.0",
        manifest=sample_manifest,
        plugin_class=MagicMock,
        source=PluginSource.BUILTIN,
    )


@pytest.fixture()
def sample_record_external(sample_manifest_no_hints):
    """Create a sample external plugin record."""
    return PluginRecord(
        plugin_type="chunker",
        plugin_id="minimal-plugin",
        plugin_version="0.1.0",
        manifest=sample_manifest_no_hints,
        plugin_class=MagicMock,
        source=PluginSource.EXTERNAL,
    )


class TestListPluginsTool:
    """Tests for ListPluginsTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = ListPluginsTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "list_plugins"
        assert "plugin_type" in schema["function"]["parameters"]["properties"]
        assert "include_disabled" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio()
    async def test_list_all_plugins(self, sample_record, sample_record_external):
        """Test listing all plugins without filter."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [sample_record, sample_record_external]
            mock_registry.is_disabled.return_value = False
            mock_registry.list_types.return_value = ["parser", "chunker"]

            tool = ListPluginsTool(context={})
            result = await tool.execute()

            assert result["count"] == 2
            assert len(result["plugins"]) == 2
            assert result["filter"] is None
            assert "parser" in result["available_types"]
            assert "chunker" in result["available_types"]

    @pytest.mark.asyncio()
    async def test_list_plugins_by_type(self, sample_record):
        """Test listing plugins filtered by type."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [sample_record]
            mock_registry.is_disabled.return_value = False
            mock_registry.list_types.return_value = ["parser"]

            tool = ListPluginsTool(context={})
            result = await tool.execute(plugin_type="parser")

            mock_registry.list_records.assert_called_with(plugin_type="parser")
            assert result["count"] == 1
            assert result["filter"] == "parser"
            assert result["plugins"][0]["type"] == "parser"

    @pytest.mark.asyncio()
    async def test_excludes_disabled_by_default(self, sample_record):
        """Test that disabled plugins are excluded by default."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [sample_record]
            mock_registry.is_disabled.return_value = True  # Plugin is disabled
            mock_registry.list_types.return_value = ["parser"]

            tool = ListPluginsTool(context={})
            result = await tool.execute()

            assert result["count"] == 0
            assert len(result["plugins"]) == 0

    @pytest.mark.asyncio()
    async def test_includes_disabled_when_requested(self, sample_record):
        """Test that disabled plugins are included when include_disabled=True."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [sample_record]
            mock_registry.is_disabled.return_value = True
            mock_registry.list_types.return_value = ["parser"]

            tool = ListPluginsTool(context={})
            result = await tool.execute(include_disabled=True)

            assert result["count"] == 1
            assert result["plugins"][0]["is_disabled"] is True

    @pytest.mark.asyncio()
    async def test_includes_agent_hints(self, sample_record):
        """Test that agent hints are included in output."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [sample_record]
            mock_registry.is_disabled.return_value = False
            mock_registry.list_types.return_value = ["parser"]

            tool = ListPluginsTool(context={})
            result = await tool.execute()

            plugin = result["plugins"][0]
            assert "agent_hints" in plugin
            assert plugin["agent_hints"]["purpose"] == "Parse documents into text"
            assert "PDF files" in plugin["agent_hints"]["best_for"]

    @pytest.mark.asyncio()
    async def test_plugin_without_hints(self, sample_record_external):
        """Test that plugins without hints don't include agent_hints field."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [sample_record_external]
            mock_registry.is_disabled.return_value = False
            mock_registry.list_types.return_value = ["chunker"]

            tool = ListPluginsTool(context={})
            result = await tool.execute()

            plugin = result["plugins"][0]
            assert "agent_hints" not in plugin

    @pytest.mark.asyncio()
    async def test_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.list_records.side_effect = Exception("Registry error")

            tool = ListPluginsTool(context={})
            result = await tool.execute()

            assert "error" in result
            assert result["count"] == 0


class TestGetPluginDetailsTool:
    """Tests for GetPluginDetailsTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = GetPluginDetailsTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_plugin_details"
        assert "plugin_id" in schema["function"]["parameters"]["properties"]
        assert "plugin_id" in schema["function"]["parameters"]["required"]

    @pytest.mark.asyncio()
    async def test_get_existing_plugin(self, sample_record):
        """Test getting details for an existing plugin."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = sample_record
            mock_registry.is_disabled.return_value = False

            tool = GetPluginDetailsTool(context={})
            result = await tool.execute(plugin_id="test-parser")

            assert result["found"] is True
            assert result["id"] == "test-parser"
            assert result["type"] == "parser"
            assert result["display_name"] == "Test Parser"
            assert result["author"] == "Test Author"
            assert result["source"] == "builtin"
            assert result["capabilities"]["supports_pdf"] is True

    @pytest.mark.asyncio()
    async def test_get_plugin_with_hints(self, sample_record):
        """Test that agent hints are included in details."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = sample_record
            mock_registry.is_disabled.return_value = False

            tool = GetPluginDetailsTool(context={})
            result = await tool.execute(plugin_id="test-parser")

            assert "agent_hints" in result
            hints = result["agent_hints"]
            assert hints["purpose"] == "Parse documents into text"
            assert hints["output_type"] == "text"
            assert "PDF files" in hints["best_for"]

    @pytest.mark.asyncio()
    async def test_get_nonexistent_plugin(self):
        """Test getting details for a non-existent plugin."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = None
            mock_registry.list_ids.return_value = ["plugin-a", "plugin-b"]

            tool = GetPluginDetailsTool(context={})
            result = await tool.execute(plugin_id="nonexistent")

            assert result["found"] is False
            assert "not found" in result["error"]
            assert "plugin-a" in result["available_plugins"]

    @pytest.mark.asyncio()
    async def test_handles_exception(self):
        """Test that exceptions are handled gracefully."""
        with patch("webui.services.agent.tools.plugins.plugin_registry") as mock_registry:
            mock_registry.find_by_id.side_effect = Exception("Lookup error")

            tool = GetPluginDetailsTool(context={})
            result = await tool.execute(plugin_id="test")

            assert result["found"] is False
            assert "error" in result
