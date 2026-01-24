"""Unit tests for pipeline management tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode
from webui.services.agent.tools.pipeline import (
    ApplyPipelineTool,
    BuildPipelineTool,
    GetPipelineStateTool,
)


@pytest.fixture()
def sample_pipeline_config():
    """Create a sample pipeline configuration dict."""
    return {
        "id": "test-pipeline",
        "version": "1.0",
        "nodes": [
            {"id": "parser", "type": "parser", "plugin_id": "text-parser", "config": {}},
            {"id": "chunker", "type": "chunker", "plugin_id": "semantic-chunker", "config": {"max_tokens": 512}},
            {"id": "embedder", "type": "embedder", "plugin_id": "dense-embedding", "config": {"model": "BAAI/bge-base-en-v1.5"}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "parser", "when": None},
            {"from_node": "parser", "to_node": "chunker", "when": None},
            {"from_node": "chunker", "to_node": "embedder", "when": None},
        ],
    }


@pytest.fixture()
def sample_template():
    """Create a sample pipeline template."""
    return PipelineTemplate(
        id="academic-papers",
        name="Academic Papers",
        description="Template for academic papers",
        suggested_for=["PDF", "research"],
        pipeline=PipelineDAG(
            id="academic-papers",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text-parser"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="semantic-chunker", config={"max_tokens": 512}),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-embedding"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        ),
        tunable=[
            TunableParameter(
                path="nodes.chunker.config.max_tokens",
                description="Max tokens per chunk",
                default=512,
                range=(128, 2048),
            ),
        ],
    )


@pytest.fixture()
def mock_conversation(sample_pipeline_config):
    """Create a mock conversation with pipeline."""
    conversation = MagicMock()
    conversation.id = "conv-123"
    conversation.current_pipeline = sample_pipeline_config
    conversation.source_id = 1
    return conversation


@pytest.fixture()
def mock_conversation_no_pipeline():
    """Create a mock conversation without pipeline."""
    conversation = MagicMock()
    conversation.id = "conv-456"
    conversation.current_pipeline = None
    conversation.source_id = None
    return conversation


class TestGetPipelineStateTool:
    """Tests for GetPipelineStateTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = GetPipelineStateTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_pipeline_state"
        assert "validate" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio()
    async def test_get_state_no_conversation(self):
        """Test getting state when no conversation in context."""
        tool = GetPipelineStateTool(context={})
        result = await tool.execute()

        assert result["has_pipeline"] is False
        assert "error" in result

    @pytest.mark.asyncio()
    async def test_get_state_no_pipeline(self, mock_conversation_no_pipeline):
        """Test getting state when conversation has no pipeline."""
        tool = GetPipelineStateTool(context={"conversation": mock_conversation_no_pipeline})
        result = await tool.execute()

        assert result["has_pipeline"] is False
        assert result["pipeline"] is None
        assert "build_pipeline" in result["message"]

    @pytest.mark.asyncio()
    async def test_get_state_with_pipeline(self, mock_conversation, sample_pipeline_config):
        """Test getting state when pipeline exists."""
        tool = GetPipelineStateTool(context={"conversation": mock_conversation})
        result = await tool.execute()

        assert result["has_pipeline"] is True
        assert result["pipeline"] == sample_pipeline_config

    @pytest.mark.asyncio()
    async def test_get_state_with_validation(self, mock_conversation):
        """Test getting state with validation enabled."""
        with patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry:
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]

            tool = GetPipelineStateTool(context={"conversation": mock_conversation})
            result = await tool.execute(validate=True)

            assert "validation" in result
            assert result["validation"]["is_valid"] is True
            assert result["validation"]["errors"] == []

    @pytest.mark.asyncio()
    async def test_get_state_validation_fails(self, mock_conversation):
        """Test getting state when validation finds errors."""
        # Create pipeline with unknown plugin
        mock_conversation.current_pipeline["nodes"][0]["plugin_id"] = "unknown-parser"

        with patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry:
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]

            tool = GetPipelineStateTool(context={"conversation": mock_conversation})
            result = await tool.execute(validate=True)

            assert result["validation"]["is_valid"] is False
            assert len(result["validation"]["errors"]) > 0


class TestBuildPipelineTool:
    """Tests for BuildPipelineTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = BuildPipelineTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "build_pipeline"
        assert "template_id" in schema["function"]["parameters"]["properties"]
        assert "nodes" in schema["function"]["parameters"]["properties"]
        assert "edges" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio()
    async def test_build_no_conversation(self):
        """Test building when no conversation in context."""
        tool = BuildPipelineTool(context={})
        result = await tool.execute(template_id="test")

        assert result["success"] is False
        assert "conversation" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_build_from_template(self, mock_conversation_no_pipeline, sample_template):
        """Test building pipeline from template."""
        with (
            patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry,
            patch("shared.pipeline.templates.load_template") as mock_load,
        ):
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]
            mock_load.return_value = sample_template

            tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
            result = await tool.execute(template_id="academic-papers")

            assert result["success"] is True
            assert result["pipeline"]["id"] == "academic-papers"
            assert result["validation"]["is_valid"] is True

    @pytest.mark.asyncio()
    async def test_build_from_template_with_tunables(self, mock_conversation_no_pipeline, sample_template):
        """Test building pipeline with tunable overrides."""
        with (
            patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry,
            patch("shared.pipeline.templates.load_template") as mock_load,
            patch("shared.pipeline.templates.resolve_tunable_path") as mock_resolve,
        ):
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]
            mock_load.return_value = sample_template

            # Mock resolve_tunable_path to return the node
            chunker_node = sample_template.pipeline.nodes[1]
            mock_resolve.return_value = (chunker_node, "max_tokens")

            tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
            result = await tool.execute(
                template_id="academic-papers",
                tunable_values={"nodes.chunker.config.max_tokens": 1024},
            )

            assert result["success"] is True
            # The tunable value should be applied
            chunker = next(n for n in result["pipeline"]["nodes"] if n["id"] == "chunker")
            assert chunker["config"]["max_tokens"] == 1024

    @pytest.mark.asyncio()
    async def test_build_from_template_invalid_tunable(self, mock_conversation_no_pipeline, sample_template):
        """Test building with invalid tunable path."""
        with (
            patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry,
            patch("shared.pipeline.templates.load_template") as mock_load,
            patch("shared.pipeline.templates.resolve_tunable_path") as mock_resolve,
        ):
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]
            mock_load.return_value = sample_template
            mock_resolve.return_value = (None, None)  # Invalid path

            tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
            result = await tool.execute(
                template_id="academic-papers",
                tunable_values={"nodes.invalid.config.value": 100},
            )

            assert result["success"] is False
            assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_build_template_not_found(self, mock_conversation_no_pipeline, sample_template):
        """Test building with non-existent template."""
        with (
            patch("shared.pipeline.templates.load_template") as mock_load,
            patch("shared.pipeline.templates.list_templates") as mock_list,
        ):
            mock_load.return_value = None
            mock_list.return_value = [sample_template]

            tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
            result = await tool.execute(template_id="nonexistent")

            assert result["success"] is False
            assert "not found" in result["error"]
            assert "academic-papers" in result["available_templates"]

    @pytest.mark.asyncio()
    async def test_build_from_custom_nodes(self, mock_conversation_no_pipeline):
        """Test building pipeline from custom nodes and edges."""
        with patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry:
            mock_registry.list_ids.return_value = ["custom-parser", "custom-chunker", "custom-embedder"]

            nodes = [
                {"id": "parser", "type": "parser", "plugin_id": "custom-parser"},
                {"id": "chunker", "type": "chunker", "plugin_id": "custom-chunker", "config": {"size": 256}},
                {"id": "embedder", "type": "embedder", "plugin_id": "custom-embedder"},
            ]
            edges = [
                {"from_node": "_source", "to_node": "parser"},
                {"from_node": "parser", "to_node": "chunker"},
                {"from_node": "chunker", "to_node": "embedder"},
            ]

            tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
            result = await tool.execute(nodes=nodes, edges=edges, pipeline_id="my-custom")

            assert result["success"] is True
            assert result["pipeline"]["id"] == "my-custom"
            assert len(result["pipeline"]["nodes"]) == 3

    @pytest.mark.asyncio()
    async def test_build_custom_missing_edges(self, mock_conversation_no_pipeline):
        """Test building custom pipeline without edges."""
        nodes = [
            {"id": "parser", "type": "parser", "plugin_id": "text-parser"},
        ]

        tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
        result = await tool.execute(nodes=nodes)

        assert result["success"] is False
        assert "edges" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_build_neither_template_nor_nodes(self, mock_conversation_no_pipeline):
        """Test building without template or nodes."""
        tool = BuildPipelineTool(context={"conversation": mock_conversation_no_pipeline})
        result = await tool.execute()

        assert result["success"] is False
        assert "template_id" in result["error"] or "nodes" in result["error"]


class TestApplyPipelineTool:
    """Tests for ApplyPipelineTool."""

    def test_schema_is_valid(self):
        """Test that the tool schema is properly defined."""
        tool = ApplyPipelineTool(context={})
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "apply_pipeline"
        assert "collection_name" in schema["function"]["parameters"]["properties"]
        assert "collection_name" in schema["function"]["parameters"]["required"]
        assert "force" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio()
    async def test_apply_no_conversation(self):
        """Test applying when no conversation in context."""
        tool = ApplyPipelineTool(context={})
        result = await tool.execute(collection_name="test")

        assert result["success"] is False
        assert "conversation" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_apply_no_session(self, mock_conversation):
        """Test applying when no session in context."""
        tool = ApplyPipelineTool(context={"conversation": mock_conversation})
        result = await tool.execute(collection_name="test")

        assert result["success"] is False
        assert "session" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_apply_no_user_id(self, mock_conversation):
        """Test applying when no user_id in context."""
        mock_session = AsyncMock()
        tool = ApplyPipelineTool(context={"conversation": mock_conversation, "session": mock_session})
        result = await tool.execute(collection_name="test")

        assert result["success"] is False
        assert "user" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_apply_no_pipeline(self, mock_conversation_no_pipeline):
        """Test applying when no pipeline configured."""
        mock_session = AsyncMock()
        tool = ApplyPipelineTool(
            context={"conversation": mock_conversation_no_pipeline, "session": mock_session, "user_id": 1}
        )
        result = await tool.execute(collection_name="test")

        assert result["success"] is False
        assert "pipeline" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_apply_with_blocking_uncertainties(self, mock_conversation):
        """Test applying when blocking uncertainties exist."""
        mock_session = AsyncMock()
        mock_uncertainty = MagicMock()
        mock_uncertainty.message = "Missing parser for PDF"
        mock_uncertainty.severity.value = "blocking"

        with (
            patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry,
            patch("webui.services.agent.repository.AgentConversationRepository") as mock_repo_class,
        ):
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]
            mock_repo = AsyncMock()
            mock_repo.get_blocking_uncertainties.return_value = [mock_uncertainty]
            mock_repo_class.return_value = mock_repo

            tool = ApplyPipelineTool(
                context={"conversation": mock_conversation, "session": mock_session, "user_id": 1}
            )
            result = await tool.execute(collection_name="test")

            assert result["success"] is False
            assert "blocking" in result["error"].lower()
            assert len(result["blocking_uncertainties"]) == 1

    @pytest.mark.asyncio()
    async def test_apply_with_force(self, mock_conversation):
        """Test applying with force=True bypasses blocking uncertainties."""
        mock_session = AsyncMock()
        mock_uncertainty = MagicMock()
        mock_uncertainty.message = "Missing parser for PDF"
        mock_uncertainty.severity.value = "blocking"

        with (
            patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry,
            patch("webui.services.factory.create_collection_service") as mock_factory,
            patch("webui.services.agent.repository.AgentConversationRepository") as mock_repo_class,
        ):
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]

            # Mock collection service
            mock_service = AsyncMock()
            mock_service.create_collection.return_value = (
                {"id": "coll-123", "name": "test"},
                {"id": "op-123"},
            )
            mock_factory.return_value = mock_service

            # Mock repository
            mock_repo = AsyncMock()
            mock_repo.set_collection.return_value = mock_conversation
            mock_repo_class.return_value = mock_repo

            tool = ApplyPipelineTool(
                context={"conversation": mock_conversation, "session": mock_session, "user_id": 1}
            )
            result = await tool.execute(collection_name="test", force=True)

            assert result["success"] is True
            assert result["collection_id"] == "coll-123"

    @pytest.mark.asyncio()
    async def test_apply_success(self, mock_conversation):
        """Test successful pipeline application."""
        mock_session = AsyncMock()

        with (
            patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry,
            patch("webui.services.factory.create_collection_service") as mock_factory,
            patch("webui.services.agent.repository.AgentConversationRepository") as mock_repo_class,
        ):
            mock_registry.list_ids.return_value = ["text-parser", "semantic-chunker", "dense-embedding"]

            # Mock collection service
            mock_service = AsyncMock()
            mock_service.create_collection.return_value = (
                {"id": "coll-123", "name": "My Collection"},
                {"id": "op-456"},
            )
            mock_factory.return_value = mock_service

            # Mock repository - no blocking uncertainties
            mock_repo = AsyncMock()
            mock_repo.get_blocking_uncertainties.return_value = []
            mock_repo.set_collection.return_value = mock_conversation
            mock_repo_class.return_value = mock_repo

            tool = ApplyPipelineTool(
                context={"conversation": mock_conversation, "session": mock_session, "user_id": 1}
            )
            result = await tool.execute(collection_name="My Collection", collection_description="Test desc")

            assert result["success"] is True
            assert result["collection_id"] == "coll-123"
            assert result["collection_name"] == "My Collection"
            assert result["operation_id"] == "op-456"
            assert result["status"] == "indexing"

            # Verify collection service was called correctly
            mock_service.create_collection.assert_called_once()
            call_args = mock_service.create_collection.call_args
            assert call_args.kwargs["name"] == "My Collection"
            assert call_args.kwargs["description"] == "Test desc"

    @pytest.mark.asyncio()
    async def test_apply_validation_error(self, mock_conversation):
        """Test applying with invalid pipeline."""
        # Create invalid pipeline (missing embedder)
        mock_conversation.current_pipeline = {
            "id": "invalid",
            "version": "1.0",
            "nodes": [
                {"id": "parser", "type": "parser", "plugin_id": "text-parser", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser", "when": None},
            ],
        }

        mock_session = AsyncMock()

        with patch("webui.services.agent.tools.pipeline.plugin_registry") as mock_registry:
            mock_registry.list_ids.return_value = ["text-parser"]

            tool = ApplyPipelineTool(
                context={"conversation": mock_conversation, "session": mock_session, "user_id": 1}
            )
            result = await tool.execute(collection_name="test")

            assert result["success"] is False
            assert "validation" in result["error"].lower()
            assert "validation_errors" in result
